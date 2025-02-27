from os import environ

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

SQLALCHEMY_DATABASE_URL = f"postgresql://{environ.get('DB_USER', "test")}:{environ.get('DB_PASSWORD', "test")}@{environ.get('DB_URL', "localhost")}:{environ.get('DB_PORT', "5432")}/{environ.get('DB_NAME', "test")}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Base(DeclarativeBase):
    pass
