import helper_libs
logger = helper_libs.logger.create_logger_for('libs')
logger('Load libs')

from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests



#url = "img.jpg"
#image = Image.open(requests.get(url, stream=True).raw)

def loadModel():
    logger('Load Model')
    processor = DPTImageProcessor.from_pretrained("Intel_dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel_dpt-large")
    return processor,model

def prepocess(name_image,model):
    logger('Proc Img')
    processor,model=model
    image = Image.open(name_image)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save(f'depth_{name_image}')


model=loadModel()

prepocess(name,model)