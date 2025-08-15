from pydantic import BaseModel
from typing import Dict

class Response(BaseModel):
    structured_data: Dict
    model: str = None
    method: str
    usage: dict = None