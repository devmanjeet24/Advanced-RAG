from fastapi import APIRouter, Depends
from app.core.dependencies import get_user

router = APIRouter()

@router.get("/me")
async def get_profile(user=Depends(get_user)):
    return {
        "email": user["email"],
        "username": user.get("username"),
        "avatar": user.get("avatar")
    }