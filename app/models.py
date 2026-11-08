from typing import Literal

from pydantic import BaseModel


class DocumentPage(BaseModel):
    document_id: str
    page_number: int
    text: str
    extraction_method: Literal["text", "ocr", "empty"]