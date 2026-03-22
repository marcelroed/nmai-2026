from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
LOG_DIR = DATA_DIR / "parsed_logs_v2"
FILES_DIR = DATA_DIR / "files"
RAW_DIR = FILES_DIR / "raw"
PARSED_DIR = FILES_DIR / "parsed"
MANIFEST_PATH = FILES_DIR / "manifest.json"

ATTACHMENT_PATTERN = re.compile(
    r"filename='(?P<filename>[^']+)', "
    r"content_base64='(?P<content_base64>[^']+)', "
    r"mime_type='(?P<mime_type>[^']+)'"
)


@dataclass(frozen=True, slots=True)
class Attachment:
    request_id: str
    filename: str
    mime_type: str
    content_bytes: bytes

    @property
    def basename(self) -> str:
        return Path(self.filename).name

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content_bytes).hexdigest()

    @property
    def is_pdf(self) -> bool:
        return self.basename.lower().endswith(".pdf")


def iter_log_files() -> list[Path]:
    return sorted(LOG_DIR.glob("*.json"))


def extract_attachments_from_payload(payload: str, request_id: str) -> list[Attachment]:
    attachments: list[Attachment] = []

    for match in ATTACHMENT_PATTERN.finditer(payload):
        content_bytes = base64.b64decode(match.group("content_base64"))
        attachments.append(
            Attachment(
                request_id=request_id,
                filename=match.group("filename"),
                mime_type=match.group("mime_type"),
                content_bytes=content_bytes,
            )
        )

    return attachments


def collect_attachments() -> list[Attachment]:
    attachments_by_name: dict[str, Attachment] = {}

    for log_path in iter_log_files():
        entries = json.loads(log_path.read_text())

        for entry in entries:
            request_id = entry.get("request_id")
            extra = entry.get("extra") or {}
            payload = extra.get("payload")
            if not isinstance(request_id, str) or not isinstance(payload, str):
                continue

            for attachment in extract_attachments_from_payload(payload, request_id):
                existing = attachments_by_name.get(attachment.basename)
                if existing is None:
                    attachments_by_name[attachment.basename] = attachment
                    continue

                if existing.sha256 != attachment.sha256:
                    msg = (
                        f"Conflicting attachment contents for {attachment.basename!r}: "
                        f"{existing.request_id} vs {attachment.request_id}"
                    )
                    raise ValueError(msg)

    return sorted(attachments_by_name.values(), key=lambda item: item.basename)


def write_raw_attachment(attachment: Attachment) -> Path:
    output_path = RAW_DIR / attachment.basename
    output_path.write_bytes(attachment.content_bytes)
    return output_path


def extract_text_from_pdf_path(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip() + "\n"


def write_parsed_pdf_text(raw_path: Path, attachment: Attachment) -> Path:
    output_path = PARSED_DIR / f"{Path(attachment.basename).stem}.txt"
    text = extract_text_from_pdf_path(raw_path)
    output_path.write_text(text)
    return output_path


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    attachments = collect_attachments()
    manifest: list[dict[str, str | None]] = []

    for attachment in attachments:
        raw_path = write_raw_attachment(attachment)
        parsed_path: Path | None = None

        if attachment.is_pdf:
            parsed_path = write_parsed_pdf_text(raw_path, attachment)

        manifest.append(
            {
                "request_id": attachment.request_id,
                "filename": attachment.filename,
                "basename": attachment.basename,
                "mime_type": attachment.mime_type,
                "sha256": attachment.sha256,
                "raw_path": str(raw_path),
                "parsed_path": str(parsed_path) if parsed_path is not None else None,
            }
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    pdf_count = sum(1 for attachment in attachments if attachment.is_pdf)
    csv_count = sum(1 for attachment in attachments if attachment.basename.lower().endswith(".csv"))
    print(f"Extracted {len(attachments)} files to {RAW_DIR}")
    print(f"Wrote {pdf_count} parsed PDF text files to {PARSED_DIR}")
    print(f"CSV files: {csv_count}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
