from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app import models, schemas


def get_centroids(db: Session, cluster_id) -> list[schemas.Point]:
    query = select(
        distinct(models.Point.cluster_id),
        func.ST_X(models.Point.centroid).label("centroid_x"),
        func.ST_Y(models.Point.centroid).label("centroid_y"),
        models.Point.radius.label("radius"),
    )
    if cluster_id:
        query = query.where(models.Point.cluster_id == cluster_id)
    out = db.execute(query.limit(100)).tuples().all()
    points = []
    for i in out:
        points.append(schemas.Centroid(cluster_id=i[0], centroid_x=i[1], centroid_y=i[2], radius=i[3]))
    return points
