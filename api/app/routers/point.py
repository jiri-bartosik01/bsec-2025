from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import crud, schemas
from app.services.database import get_db

router = APIRouter(
    tags=["point"],
    prefix="/points",
)


@router.get("", response_model=list[schemas.Point])
async def read_points(
    db: Annotated[Session, Depends(get_db)],
    cluster_id: int | None = None,
):
    return crud.get_points(db, cluster_id)
