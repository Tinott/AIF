import io
import os

import gradio as gr
import numpy as np
import requests
from PIL import Image

API_URL = os.environ.get("API_URL", "http://127.0.0.1:5075")


def to_mnist(sketch):
    img = sketch["composite"] if isinstance(sketch, dict) else sketch
    if img is None:
        return None
    arr = np.array(img)
    # trait opaque sur fond transparent -> on prend le canal alpha
    if arr.ndim == 3 and arr.shape[2] == 4 and arr[..., 3].min() < 255:
        digit = arr[..., 3]
    else:
        # trait noir sur fond blanc -> on inverse (MNIST = chiffre blanc sur fond noir)
        digit = 255 - np.array(Image.fromarray(arr).convert("L"))
    return Image.fromarray(digit.astype(np.uint8)).resize((28, 28))


def recognize_digit(sketch):
    if sketch is None:
        return None
    img = to_mnist(sketch)
    if img is None:
        return None
    img_binary = io.BytesIO()
    img.save(img_binary, format="PNG")
    response = requests.post(f"{API_URL}/predict", data=img_binary.getvalue())
    return str(response.json()["prediction"])


if __name__ == "__main__":
    gr.Interface(
        fn=recognize_digit,
        inputs="sketchpad",
        outputs="label",
        live=True,
        description="Draw a number on the sketchpad to see the model's prediction.",
    ).launch(server_name="0.0.0.0", server_port=7860)