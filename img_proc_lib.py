import helper_libs
logger = helper_libs.logger.create_logger_for('libs')
logger('Load libs')


from PIL import ImageEnhance, Image
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn



# CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LS
def ls(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


# Shadow remover
class ShadowModel(nn.Module):
    def __init__(self, num_iterations=2):
        super(ShadowModel, self).__init__()
        self.num_iterations = num_iterations
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for _ in range(self.num_iterations):
            x = self.restore(x)
        return x

    def restore(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x



class IMGProc:
    def __init__(self):
        # Инициализация устройства
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Инициализация преобразования
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        pass

    def _to_device(self, model):
        model.to(self.device).eval()
        return model

    def shadow(self, img, model=[], value=1.0):
        
        
        img_tensor = self.transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out_tensor = model(img_tensor)
        
        processed_img = transforms.ToPILImage()(out_tensor.cpu().squeeze(0))
        
        # Интерполирование между исходным и обработанным изображением
        out_img = Image.blend(img, processed_img, value)
        
        return out_img

    def adjust_contrast(self, img, value=1.0):
        enhancer = ImageEnhance.Contrast(img)
        out_img = enhancer.enhance(value)
        return out_img

    def adjust_brightness(self, img, value=1.0):
        enhancer = ImageEnhance.Brightness(img)
        out_img = enhancer.enhance(value)
        return out_img

    def adjust_exposure(self, img, value=1.0):
        enhancer = ImageEnhance.Brightness(img)
        out_img = enhancer.enhance(value)
        return out_img

    def adjust_hue(self, img, value=0):
        img_np = np.array(img)
        hsv_img = np.array(img.convert('HSV'))

        hsv_img[..., 0] = (hsv_img[..., 0] + int(value*255)) % 256

        out_img = Image.fromarray(hsv_img, 'HSV').convert('RGB')
        return out_img
