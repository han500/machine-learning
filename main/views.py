from django.shortcuts import render
import base64
import io
import os

import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
from django.templatetags.static import static

from .forms import InputForm
# Create your views here.


model = models.efficientnet_b0()
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=1, bias=True),
)

model.load_state_dict(torch.load('./static/main/B0.pth', map_location=torch.device('cpu')))
model.eval()


def apply_transformations(image_bytes):
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transformations(image).unsqueeze(0)


def predict(image_bytes):
    tensor_of_image = apply_transformations(image_bytes)
    sigmoid_layer = torch.nn.Sigmoid()
    outputs = sigmoid_layer(model.forward(tensor_of_image)).round()
    if outputs.item() == 0.0:
        predicted_label = 'Normal'
    else:
        predicted_label = 'Pneumonia'

    return predicted_label


def home(request):
    image_uri = None
    predicted_label = None

    if request.method == "POST":
        form = InputForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data["image"]
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            predicted_label = predict(image_bytes)


    else:
        form = InputForm()

    
    return render(request, 'main/image.html', {"form": form, "image_uri": image_uri, "predicted_label": predicted_label})