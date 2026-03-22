from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

from pypdf import PdfReader


class PdfInputFile(Protocol):
    filename: str
    content_base64: str
    mime_type: str


@dataclass(frozen=True, slots=True)
class ExamplePdfInputFile:
    filename: str
    content_base64: str
    mime_type: str


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from a PDF's embedded text layer."""
    if not pdf_bytes:
        msg = "PDF content is empty."
        raise ValueError(msg)

    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    page_texts: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        cleaned_text = text.strip()
        if cleaned_text:
            page_texts.append(cleaned_text)

    return "\n\n".join(page_texts)


def extract_text_from_pdf_base64(content_base64: str) -> str:
    try:
        normalized_content = "".join(content_base64.split())
        pdf_bytes = b64decode(normalized_content, validate=True)
    except BinasciiError as exc:
        msg = "PDF content is not valid base64."
        raise ValueError(msg) from exc

    return extract_text_from_pdf_bytes(pdf_bytes)


def extract_text_from_input_file(input_file: PdfInputFile) -> str:
    if input_file.mime_type != "application/pdf":
        msg = (
            f"Expected application/pdf for {input_file.filename!r}, "
            f"got {input_file.mime_type!r}."
        )
        raise ValueError(msg)

    return extract_text_from_pdf_base64(input_file.content_base64)


if __name__ == "__main__":
    data = ExamplePdfInputFile(
        filename="files/tilbudsbrev_pt_08.pdf",
        content_base64="JVBERi0xLjMKJenr8b8KMSAwIG9iago8PAovQ291bnQgMQovS2lkcyBbMyAwIFJdCi9NZWRpYUJveCBbMCAwIDU5NS4yOCA4NDEuODldCi9UeXBlIC9QYWdlcwo+PgplbmRvYmoKMiAwIG9iago8PAovT3BlbkFjdGlvbiBbMyAwIFIgL0ZpdEggbnVsbF0KL1BhZ2VMYXlvdXQgL09uZUNvbHVtbgovUGFnZXMgMSAwIFIKL1R5cGUgL0NhdGFsb2cKPj4KZW5kb2JqCjMgMCBvYmoKPDwKL0NvbnRlbnRzIDQgMCBSCi9QYXJlbnQgMSAwIFIKL1Jlc291cmNlcyA3IDAgUgovVHlwZSAvUGFnZQo+PgplbmRvYmoKNCAwIG9iago8PAovRmlsdGVyIC9GbGF0ZURlY29kZQovTGVuZ3RoIDYwMQo+PgpzdHJlYW0KeJyNVNtuEzEQfe9XzAsSSOB677t9o2oDhYZbAq/IkScbJ44dbNOKv2e8m3SbpqBdKfLGO2fOGXvOpPDhjLOigvuzyzmcTxJIasY5zJdwPY9bKc9ZkkPVJHGZS3g5v7m9/H4Fn6cwo9fbm0/vXsF8vQ8/n6SQJEcZMkLWUGUVa8ouwce1QIcwFU4JI2AmTLD+9aMkewQvWVV3iB8KVsJBq1GiAXEHAoLSiz8gsQVPr1qZlr54u4Uvzvo1bkKMdaAoWmL8DLeiRceOtFK12TNay7JgvGe+wiD0Gt1T2HMlllnOCt7BJlZ61F6KYC/+fzpJlbImeYyNB56ypGnyUaS0W6f9xSgdHFWtPY4kHbCcAgqW8rQcQ1rUzaEb3hqPIUROv7RuO474EX4ifHi4wlHcZcnyvpFme5jf0Z2jCSPJhwQJ54y/GMWaZ6xq+oqd19aYkWQDrkw5PbAZ1UtFSkuy51ugknREciTlAK1YQUbZkg929JOifZqADwl+wQN5XtWs7kRzlqcUcB8tqDfCXcDxfa+slhjAYftbk6d9581e71bptdX2jmxpW1igdGoZ0PgoxVsj9EoYubAbBuQxaNeoo18pmi7zDqla2Ak4dCN8PdJXUt92/sxYUeedvq0wneG3KIFaS0az+5jWeCVVC3a386rtdTvlA4NrKsM9sBEx/XsaRbsZ7FMzmKBTSNUMogZJec7qqjv0ZYzaIQ0kB0uruyVudWfBTsZcRpOV9yacknaKMZr0rhSJMKdTMSsSlmVd+M9/P6cwWvK+D99/e3OYiWgOgX8BhF+epwplbmRzdHJlYW0KZW5kb2JqCjUgMCBvYmoKPDwKL0Jhc2VGb250IC9IZWx2ZXRpY2EtQm9sZAovRW5jb2RpbmcgL1dpbkFuc2lFbmNvZGluZwovU3VidHlwZSAvVHlwZTEKL1R5cGUgL0ZvbnQKPj4KZW5kb2JqCjYgMCBvYmoKPDwKL0Jhc2VGb250IC9IZWx2ZXRpY2EKL0VuY29kaW5nIC9XaW5BbnNpRW5jb2RpbmcKL1N1YnR5cGUgL1R5cGUxCi9UeXBlIC9Gb250Cj4+CmVuZG9iago3IDAgb2JqCjw8Ci9Gb250IDw8L0YxIDUgMCBSCi9GMiA2IDAgUj4+Ci9Qcm9jU2V0IFsvUERGIC9UZXh0IC9JbWFnZUIgL0ltYWdlQyAvSW1hZ2VJXQo+PgplbmRvYmoKOCAwIG9iago8PAovQ3JlYXRpb25EYXRlIChEOjIwMjYwMzA0MTkxODA2WikKPj4KZW5kb2JqCnhyZWYKMCA5CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAxNSAwMDAwMCBuIAowMDAwMDAwMTAyIDAwMDAwIG4gCjAwMDAwMDAyMDUgMDAwMDAgbiAKMDAwMDAwMDI4NSAwMDAwMCBuIAowMDAwMDAwOTU4IDAwMDAwIG4gCjAwMDAwMDEwNjAgMDAwMDAgbi AKMDAwMDAwMTE1NyAwMDAwMCBuIAowMDAwMDAxMjU0IDAwMDAwIG4gCnRyYWlsZXIKPDwKL1NpemUgOQovUm9vdCAyIDAgUgovSW5mbyA4IDAgUgovSUQgWzxCNTlDRkNCQzY3NTY5RTEyMzg5Mjg3OEExM0RGQTVCMT48QjU5Q0ZDQkM2NzU2OUUxMjM4OTI4NzhBMTNERkE1QjE+XQo+PgpzdGFydHhyZWYKMTMwOQolJUVPRgo=",
        mime_type="application/pdf",
    )

    text = extract_text_from_input_file(data)
    print(text)
