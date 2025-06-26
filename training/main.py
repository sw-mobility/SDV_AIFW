from fastapi import FastAPI
from routers.yolo_router import yolo_router

app = FastAPI()
app.include_router(yolo_router)
