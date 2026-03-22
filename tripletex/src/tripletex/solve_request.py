from pydantic import BaseModel

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import TripletexCredentials


class InputFile(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class SolveRequest(BaseModel):
    prompt: str
    files: list[InputFile]
    tripletex_credentials: TripletexCredentials

    def to_tripletex_client(self) -> LoggedHTTPClient:
        return self.tripletex_credentials.to_client()
