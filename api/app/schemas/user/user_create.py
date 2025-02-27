from pydantic import BaseModel


class UserCreate(BaseModel):
    first_name: str
    last_name: str
    username: str
    email: str | None
    age: int | None
