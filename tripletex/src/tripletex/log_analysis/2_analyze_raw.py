import json
from ast import literal_eval
from pathlib import Path

files = list((Path("data") / "parsed_logs_v2").glob("*"))
literal_eval(json.loads(Path(files[0]).read_text())[-1]["extra"]["body"]["preview"])[
    "prompt"
]
