from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import crud, schemas
from app.services.database import get_db

router = APIRouter(
    tags=["centroid"],
    prefix="/centroids",
)


@router.get("", response_model=list[schemas.Centroid])
async def read_centroids(
    db: Annotated[Session, Depends(get_db)],
    cluster_id: int | None = None,
):
    return crud.get_centroids(db, cluster_id)
