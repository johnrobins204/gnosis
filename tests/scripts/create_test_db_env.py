import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sqlalchemy import create_engine
from db_migrations.models import Base

# Debug: print environment variables
print("[DEBUG] DB_USER:", os.getenv("DB_USER"))
print("[DEBUG] DB_PASSWORD:", "***" if os.getenv("DB_PASSWORD") else None)
print("[DEBUG] DB_NAME:", os.getenv("DB_NAME"))
print("[DEBUG] DB_HOST:", os.getenv("DB_HOST", "localhost"))
print("[DEBUG] DB_PORT:", os.getenv("DB_PORT", "5432"))

# Load DB credentials from environment variables
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "5432")
db = os.getenv("DB_NAME")

if not all([user, password, db]):
    raise ValueError("Please set DB_USER, DB_PASSWORD, and DB_NAME environment variables for the test database.")

masked_conn_str = f"postgresql+psycopg2://{user}:***@{host}:{port}/{db}"
print("[DEBUG] Connection string:", masked_conn_str)
conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
engine = create_engine(conn_str)

print(f"Creating tables in test database: {db} at {host}:{port} as user {user}")
Base.metadata.create_all(engine)
print("Test database tables created successfully.")
