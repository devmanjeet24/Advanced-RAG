from fastapi import Depends
from fastapi.security import HTTPBearer
from jose import jwt
from app.core.config import settings
from app.db.mongodb import db
from bson import ObjectId

security = HTTPBearer()

async def get_user(token=Depends(security)):
    payload = jwt.decode(token.credentials, settings.SECRET_KEY, algorithms=["HS256"])
    return await db.users.find_one({"_id": ObjectId(payload["user_id"])})