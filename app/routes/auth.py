from fastapi import APIRouter, HTTPException
from app.db.schemas import UserLogin, UserCreate
from app.db.mongodb import db
from app.core.security import *

router = APIRouter()

@router.post("/register")
async def register(data: UserCreate):
    data = data.dict()

    if await db.users.find_one({"email": data["email"]}):
        raise HTTPException(400, "User exists")

    data["password"] = hash_password(data["password"])
    await db.users.insert_one(data)

    return {"msg": "registered"}

@router.post("/login")
async def login(data: UserLogin):
    data = data.dict()

    user = await db.users.find_one({"email": data["email"]})

    if not user or not verify_password(data["password"], user["password"]):
        raise HTTPException(401, "Invalid credentials")

    return {"token": create_token({"user_id": str(user["_id"])})}