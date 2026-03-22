from pathlib import Path

assert Path("pyproject.toml").read_text().splitlines()[1] == 'name = "tripletex"'
(DATA_DIR := Path("data")).mkdir(exist_ok=True)


BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SESSION_TOKEN = ""  # removed

AUTH = ("0", SESSION_TOKEN)
