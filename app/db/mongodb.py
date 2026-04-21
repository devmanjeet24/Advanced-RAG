from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client = AsyncIOMotorClient(
    settings.MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True   # 🔥 ADD THIS LINE
)

db = client[settings.DB_NAME]