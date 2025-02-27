from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import models, schemas


def get_points(db: Session, cluster_id) -> list[schemas.Point]:
    query = select(
        func.ST_X(models.Point.geometry).label("x"),
        func.ST_Y(models.Point.geometry).label("y"),
        models.Point.cluster_id.label("cluster_id"),
        func.ST_X(models.Point.centroid).label("centroid_x"),
        func.ST_Y(models.Point.centroid).label("centroid_y"),
        models.Point.radius.label("radius"),
    )
    if cluster_id:
        query = query.where(models.Point.cluster_id == cluster_id)
    out = db.execute(query.limit(100)).tuples().all()
    points = []
    for i in out:
        points.append(schemas.Point(x=i[0], y=i[1], cluster_id=i[2], centroid_x=i[3], centroid_y=i[4], radius=i[5]))
    return points
