from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME")

    SECRET_KEY = os.getenv("SECRET_KEY")
    ALGORITHM = "HS256"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Kept for backward-compat; not used in advanced pipeline
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")


settings = Settings()