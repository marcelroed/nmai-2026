#!/usr/bin/env python3

import json
import sys
import urllib.error
import urllib.request

import requests

REQUEST_BODY = {
    "prompt": "Opprett ein faktura til kunden Strandvik AS (org.nr 900314183) med tre produktlinjer: Webdesign (9716) til 13450 kr med 25 % MVA, Skylagring (6906) til 7700 kr med 15 % MVA (næringsmiddel), og Systemutvikling (2265) til 18800 kr med 0 % MVA (avgiftsfri).",
    "files": [],
    "tripletex_credentials": {
        "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
        "session_token": "",  # removed
    },
}


def resolve_endpoint(selector: str) -> str:
    if selector == "s":
        return "https://tripletex-agent-1010470531441.europe-north1.run.app/solve"
    if selector == "l":
        return "http://localhost:8000/solve"

    print(f"Unknown endpoint selector: {selector}", file=sys.stderr)
    print("Use 'l' for localhost or 's' for server.", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    selector = sys.argv[1] if len(sys.argv) > 1 else "l"
    endpoint = resolve_endpoint(selector)
    print(f"Using endpoint: {endpoint}")

    cookies = {
        "access_token": "",  # removed
    }

    auth_token = requests.get(
        "https://api.ainm.no/tripletex/my/submissions", cookies=cookies
    ).json()[1]["id"]

    payload = json.dumps(REQUEST_BODY).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer DEBUG{auth_token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as error:
        print(f"Request failed: {error.reason}", file=sys.stderr)
        return 1

    sys.stdout.write(body)
    if body and not body.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
