# FROM ultralytics/ultralytics:latest-jetson-jetpack6
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FOR ORIN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

FROM python:3.10-slim

WORKDIR /workspace1/validation

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5004"] 