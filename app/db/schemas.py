from pydantic import BaseModel, EmailStr
from typing import Optional, List

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    username: str
    
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class DocumentSchema(BaseModel):
    filename: str
    content: str
    metadata: dict

class QueryRequest(BaseModel):
    query: str
    filters: Optional[dict] = None

class EvalRequest(BaseModel):
    answer: str
    context: List[str]