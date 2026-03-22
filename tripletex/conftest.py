import os
from pathlib import Path

# Load .env file if it exists (no dependency needed)
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if not os.environ.get(key.strip()):
            os.environ[key.strip()] = value
