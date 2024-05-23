import cv2
import numpy as np
import torch
from flask import Flask, render_template, request
from model.facenetmodel import FacenetFineTuned
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

app = Flask(__name__)

model = FacenetFineTuned()
model_path = "model/facenetfinetuned.pth"
loaded_state_dict = torch.load(model_path, map_location=torch.device("cpu"))

model.load_state_dict(loaded_state_dict, strict=False)

img_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(
            device="cpu"
        ),
    ]
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":

            img = cv2.imdecode(
                np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
            )
            resized_img = cv2.resize(img, (600, 600))
            tensor_img = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
            tensor_img = tensor_img.unsqueeze(0)

            with torch.no_grad():
                prediction = model(tensor_img)

            threhold = 0.5
            if prediction.item() >= threhold:
                result = "This is a real face."
            else:
                result = "This is a fake."

            return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
