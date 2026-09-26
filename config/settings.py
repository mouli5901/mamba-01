import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Ground rules enforcement: LIVE_TRADING defaults to False
LIVE_TRADING: bool = os.getenv("LIVE_TRADING", "false").lower() in ("true", "1", "yes")

# Database URLs
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://kite:kite@localhost:5432/kite"
)
DATABASE_URL_SYNC: str = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg2://kite:kite@localhost:5432/kite"
)

# Redis URLs
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

# API Keys
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# Kite Connect
KITE_API_KEY: str = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET: str = os.getenv("KITE_API_SECRET", "")
