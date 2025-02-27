from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float

    cluster_id: int
    centroid_x: float
    centroid_y: float
    radius: float

    class Config:
        from_attributes = True
