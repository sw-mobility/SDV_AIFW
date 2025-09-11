# FROM ultralytics/ultralytics:latest-jetson-jetpack6
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FOR ORIN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

FROM python:3.10-slim

WORKDIR /workspace4/training

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# COPY packages/torch-2.3.0-cp310-cp310-linux_aarch64.whl ./
# RUN pip install --no-cache-dir torch-2.3.0-cp310-cp310-linux_aarch64.whl

# COPY packages/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl ./
# RUN pip install --no-cache-dir torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl

# COPY packages/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl ./
# RUN pip install --no-cache-dir torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5003"]
