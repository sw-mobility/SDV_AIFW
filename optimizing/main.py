from fastapi import FastAPI
from routers.optimizing_route import router as conversion_router
import uvicorn
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Optimizing Service")

app = FastAPI()

# Include the router for conversion routes
app.include_router(conversion_router)
logger.info("Router included successfully")

# Debugging PYTHONPATH and directory structure
logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
logger.info(f"Current Directory: {os.listdir('.')}")

# Entry point for the application
if __name__ == "__main__":
    logger.info("Starting Uvicorn server")
    uvicorn.run("main:app", host="0.0.0.0", port=5005, reload=False)
