from fastapi import FastAPI
from routers.yolo_route import yolo_route

app = FastAPI()
app.include_router(yolo_route)
