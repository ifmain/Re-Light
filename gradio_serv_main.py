import gradio as gr
from PIL import Image
import torch

from os.path import isfile, join
from os import listdir
import numpy as np
import random
import os

from transformers import DPTImageProcessor, DPTForDepthEstimation






import helper_libs

helper_libs.logger.rule.main(True)
helper_libs.logger.rule.libs(False)

from img_proc_lib import *

# img_proc.shadow(img, model=model, value=[float])
# img_proc.adjust_contrast(img, value=[float])
# img_proc.adjust_brightness(img, value=[float])
# img_proc.adjust_exposure(img, value=[float])
# img_proc.adjust_hue(img, value=[float])

logger = helper_libs.logger.create_logger_for('main')

logger('Load model')
model =  ShadowModel().to(device)
model.load_state_dict(torch.load('shadowRM2.pth'))
model.eval()

def loadModelDepth():
    logger('Load Model')
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    return processor,model

modelDepth=loadModelDepth()
img_proc = IMGProc()

def process_image(img_np, operation, value):
    if img_np.max() <= 1:
        img_np = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    
    if operation == "shadow":
        return img_proc.shadow(img, model=model, value=value)
    elif operation == "contrast":
        return img_proc.adjust_contrast(img, value=value)
    elif operation == "brightness":
        return img_proc.adjust_brightness(img, value=value)
    elif operation == "exposure":
        return img_proc.adjust_exposure(img, value=value)
    elif operation == "hue":
        return img_proc.adjust_hue(img, value=value)




def estimate_depth(image_array):
    logger('Proc Img')
    processor, model = modelDepth
    
    image = Image.fromarray((image_array * 255).astype('uint8'))  # Преобразование массива numpy обратно в изображение PIL
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
    return depth
    

# Create a Blocks-based Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("Process images with operations or estimate depth.")
    
    with gr.Tab("Image Processing"):
        with gr.Row():
            image_input = gr.Image()
            operation_dropdown = gr.Dropdown(choices=["shadow", "contrast", "brightness", "exposure", "hue"])
            value_slider = gr.Slider(minimum=0, maximum=2, step=0.1)
            proc_image_output = gr.Image(type='pil')
            process_button = gr.Button("Process")
        
        process_button.click(process_image, inputs=[image_input, operation_dropdown, value_slider], outputs=proc_image_output)
    
    with gr.Tab("Depth Estimation"):
        depth_image_input = gr.Image()
        depth_image_output = gr.Image(type='pil')
        depth_button = gr.Button("Estimate Depth")
        
        depth_button.click(estimate_depth, inputs=depth_image_input, outputs=depth_image_output)
    
    with gr.Accordion("Open for More!"):
        gr.Markdown("Additional info or features could be placed here.")

if __name__ == "__main__":
    demo.launch()

'''
for i in range(len(ls('v'))):
    n=(f'img{str(i+1).zfill(4)}.bmp')
    img = Image.open(f'v/{n}').convert('RGB')
    out_img=shadow(model,img,value=1.0)
    out_img.save(f'vv/{n}'.replace('.bmp','.jpg'))
'''



'''
logger('Processing')
img = Image.open('67.jpg').convert('RGB')
img_proc = IMGProc()
result = img_proc.shadow(img, model=model, value=1.0)
result.save('67_sr2.jpg')
'''