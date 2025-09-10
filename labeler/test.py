from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define remote image or video URL
source = "https://ultralytics.com/images/bus.jpg"

source = "img.txt"

# Run inference on the source
results = model(source)  # list of Results objects



print(results)



for result in results:
    result.save_txt("output.txt")
    print(result.path)
    