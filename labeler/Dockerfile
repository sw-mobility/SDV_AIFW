FROM ultralytics/ultralytics:latest-jetson-jetpack6

WORKDIR /app

COPY routers/ ./routers/
COPY services/ ./services/
COPY main.py ./
COPY models/ ./models/
COPY data/ /workspace/data/
COPY launch /launch
COPY requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt \
    && mkdir -p /export/training_dataset_output \
    && chmod -R 777 /export \
    && mkdir -p /export/labeler_output \
    && mkdir -p /workspace/data/testing_images \
    && chmod -R 777 /workspace

ENTRYPOINT ["/launch"]