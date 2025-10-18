import os
import json
import torch
import requests
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://huggingface.co/kcmandable/convnext_tiny_benthic/resolve/main/convnext_tiny_best.pth"
CLASSES_URL = "https://huggingface.co/kcmandable/convnext_tiny_benthic/resolve/main/classes.json"

MODEL_PATH = "convnext_tiny_best.pth"
CLASSES_PATH = "classes.json"

# Download if needed
for url, path in [(MODEL_URL, MODEL_PATH), (CLASSES_URL, CLASSES_PATH)]:
    if not os.path.exists(path):
        print(f"🔽 Downloading {path}...")
        r = requests.get(url)
        open(path, "wb").write(r.content)

# Load model
device = torch.device("cpu")
model = models.convnext_tiny(pretrained=False)
num_ftrs = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_ftrs, 30)  # adjust if you have a different number of classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load class names
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/")
def home():
    return jsonify({"message": "ConvNeXt classification API is running"})

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(request.files["image"].stream).convert("RGB")
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        pred_class = class_names[preds.item()]

    return jsonify({"class": pred_class})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
