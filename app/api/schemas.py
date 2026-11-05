from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

class Source(BaseModel):
    file: str
    page: int
    text: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    seconds: float