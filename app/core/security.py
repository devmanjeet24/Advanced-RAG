from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from app.core.config import settings

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p: str):
    return pwd.hash(p)

def verify_password(p: str, h: str):
    return pwd.verify(p, h)

def create_token(data: dict):
    data.update({"exp": datetime.utcnow() + timedelta(days=7)})
    return jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)