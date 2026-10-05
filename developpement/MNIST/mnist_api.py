import argparse
import io

import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image

from model import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='weights/mnist_net.pth')
args = parser.parse_args()

model = MNISTNet().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    img_pil = Image.open(io.BytesIO(request.data))
    tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        _, predicted = model(tensor).max(1)
    return jsonify({"prediction": int(predicted[0])})


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    images_binary = request.files.getlist("images[]")
    tensors = [transform(Image.open(f.stream)) for f in images_binary]
    batch_tensor = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        _, predictions = model(batch_tensor).max(1)
    return jsonify({"predictions": predictions.tolist()})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5075)