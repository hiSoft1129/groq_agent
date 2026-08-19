from sqlalchemy import create_engine

DATABASE_URL = (
    "postgresql+psycopg://postgres:"
    "Donkey1129+_@localhost:5432/chatdb"
)

engine = create_engine(DATABASE_URL)

with engine.connect() as connection:
    print("Connected to PostgreSQL!")