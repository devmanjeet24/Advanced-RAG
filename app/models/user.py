from datetime import datetime

def user_model(user):
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "username": user.get("username"),
        "avatar": user.get("avatar"),
        "created_at": user.get("created_at", datetime.utcnow())
    }