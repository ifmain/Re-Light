from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget,QDesktopWidget,QColorDialog,QFormLayout, QSlider, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt

from PIL import Image, ImageDraw
import numpy as np
import requests
import base64 
import sys
import io







def get_depth(input_image):
    # Convert PIL image to base64
    buffered = io.BytesIO()
    input_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Make the HTTP request
    response = requests.post(
        "http://127.0.0.1:7860/run/predict",
        headers={'Content-Type': 'application/json'},
        json={
            "data": [f"data:image/png;base64,{base64_image}"],
            "fn_index": 1
        }
    )

    data = response.json()
    
    # Convert the output base64 image to a PIL image
    output_image_src = data["data"][0]
    output_image_data = base64.b64decode(output_image_src.split(",")[1])
    output_image = Image.open(io.BytesIO(output_image_data))
    
    return output_image

def aply_effect(input_image,eff):
    # Convert PIL image to base64
    buffered = io.BytesIO()
    input_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Make the HTTP request
    response = requests.post(
        "http://127.0.0.1:7860/run/predict",
        headers={'Content-Type': 'application/json'},
        json={
            "data": [f"data:image/png;base64,{base64_image}",eff,1.0],
            "fn_index": 0
        }
    )

    data = response.json()
    
    # Convert the output base64 image to a PIL image
    output_image_src = data["data"][0]
    output_image_data = base64.b64decode(output_image_src.split(",")[1])
    output_image = Image.open(io.BytesIO(output_image_data))
    
    return output_image

def apply_light_effect(img, position, radius, light_color=(255, 0, 0), height_factor=1, falloff=1):
    height_factor = 4 - (4 * height_factor)

    img = img.convert('RGBA')

    # Get the value of the center point
    center_value = img.getpixel(position)[0]

    # Create a new image for the light source
    light_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(light_img)

    # Define bounds for the loop based on the light source position and radius
    x_min = max(0, position[0] - radius)
    x_max = min(img.width, position[0] + radius)
    y_min = max(0, position[1] - radius)
    y_max = min(img.height, position[1] + radius)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            distance = ((x - position[0]) ** 2 + (y - position[1]) ** 2) ** 0.5
            current_value = img.getpixel((x, y))[0]

            if distance < radius:
                alpha = int(255 * (1 - (distance / radius)**falloff))

                # Taking height_factor into account
                offset = (current_value - center_value) * height_factor

                if offset > 0:
                    alpha = int(alpha * (1 - offset / 255.0))
                elif offset < 0:
                    alpha = int(alpha * (1 + offset / 255.0))

                draw.point((x, y), fill=light_color + (alpha,))

    return light_img



def resize_image_to_screen_height(img):
    # Получить размер экрана
    screen = QDesktopWidget().screenGeometry()
    screen_height = screen.height()

    # На Windows панель задач обычно имеет высоту около 40 пикселей. 
    # Это может отличаться в зависимости от настроек и версии ОС, 
    # поэтому возможно потребуется корректировка.
    taskbar_height = 40  
    visible_screen_height = screen_height - taskbar_height
    
    # Сохраняем соотношение сторон изображения при изменении размера
    ratio = img.width / img.height
    new_width = int(visible_screen_height * ratio)
    
    return img.resize((new_width, visible_screen_height), Image.LANCZOS)

class ControlWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        self.layout = QVBoxLayout()
        
         # Radius Slider
        self.radius_label = QLabel("Radius: 400")
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setMinimum(50)
        self.radius_slider.setMaximum(800)
        self.radius_slider.setValue(400)
        self.radius_slider.valueChanged.connect(self.update_radius)
        self.layout.addWidget(self.radius_label)
        self.layout.addWidget(self.radius_slider)
        
        # Height Factor Slider
        self.height_factor_label = QLabel("Height Factor: 0.0")
        self.height_factor_slider = QSlider(Qt.Horizontal)
        self.height_factor_slider.setMinimum(0)
        self.height_factor_slider.setMaximum(100)
        self.height_factor_slider.setValue(0)
        self.height_factor_slider.valueChanged.connect(self.update_height_factor)
        self.layout.addWidget(self.height_factor_label)
        self.layout.addWidget(self.height_factor_slider)

        # Falloff Slider
        self.falloff_label = QLabel("Falloff: 2.0")
        self.falloff_slider = QSlider(Qt.Horizontal)
        self.falloff_slider.setMinimum(1)
        self.falloff_slider.setMaximum(40)
        self.falloff_slider.setValue(20)
        self.falloff_slider.valueChanged.connect(self.update_falloff)
        self.layout.addWidget(self.falloff_label)
        self.layout.addWidget(self.falloff_slider)
        
        # Button to choose light color
        self.color_button = QPushButton("Choose Light Color", self)
        self.color_button.clicked.connect(self.choose_light_color)
        self.layout.addWidget(self.color_button)

        self.setLayout(self.layout)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()
    def update_height_factor(self, value):
        self.main_window.height_factor = value * 0.01
        self.height_factor_label.setText(f"Height Factor: {self.main_window.height_factor:.2f}")

    def update_falloff(self, value):
        self.main_window.falloff = value * 0.1
        self.falloff_label.setText(f"Falloff: {self.main_window.falloff:.1f}")

    def update_radius(self, value):
        self.main_window.radius = value
        self.radius_label.setText(f"Radius: {self.main_window.radius}")
    
    def choose_light_color(self):
        color = QColorDialog.getColor()

        if color.isValid():
            self.main_window.light_color = (color.red(), color.green(), color.blue())

    

class LightApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.height_factor = 0.0
        self.falloff = 2.0
        self.radius = 400
        self.control_window = ControlWindow(self)
        self.control_window.show()
        
        self.light_color = (255, 255, 255)

        
        img1=Image.open("image.jpg")
        
        self.img = img1.convert('RGBA')
        #self.img = resize_image_to_screen_height(self.img)
        
        
        try:
            self.img2=Image.open("shadow.png").convert('RGBA')
            self.depth=Image.open("depth.png").convert('RGBA')
        except:
            self.img2=aply_effect(img1,'shadow').convert('RGBA').resize(self.img.size, Image.LANCZOS)
            self.img2.save('shadow.png')
            #self.img2 = resize_image_to_screen_height(self.img2)
            
            self.depth = get_depth(self.img).resize(self.img.size, Image.LANCZOS)
            self.depth.save('depth.png')
        
            #self.depth = resize_image_to_screen_height(self.depth)
        
        self.result_img = self.img.copy()

        # Only after all the setup, call the initUI method
        self.initUI()
    
    def multiply_blend(self, base_img, light_img, light_color):
        base_img_arr = np.array(base_img, dtype=np.float32) / 255.0
        light_img_arr = np.array(light_img, dtype=np.float32) / 255.0

        blended = np.zeros_like(base_img_arr)

        alpha_factor = light_img_arr[:, :, 3]

        for channel in range(3):  # R, G, B
            blended[:, :, channel] = base_img_arr[:, :, channel] * (1.0 + (alpha_factor * (light_color[channel] / 255.0 - 1.0)))

        # Учитываем альфа-канал света как маску
        blended[:, :, 3] = alpha_factor

        blended_img = Image.fromarray((blended * 255).astype(np.uint8), 'RGBA')
        return blended_img


    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel(self)
        pixmap = self.convert_pil_to_pixmap(self.img)
        self.label.setPixmap(pixmap)
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setWindowTitle('Interactive Light Source with PyQt')
        self.show()
    
    def update_image_effect(self):
        light_position = (self.width() // 2, self.height() // 2)
        light_img = apply_light_effect(self.depth, light_position, self.radius, height_factor=self.height_factor, falloff=self.falloff, light_color=self.light_color)
        
        # Применяем multiply смешивание к img2 с учетом цвета света
        img2_multiplied_with_light = self.multiply_blend(self.img2, light_img, self.light_color)

        # Налагаем полученное изображение на исходное
        self.result_img = Image.alpha_composite(self.img, img2_multiplied_with_light)

        pixmap = self.convert_pil_to_pixmap(self.result_img)
        self.label.setPixmap(pixmap)

    
    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            light_position = (event.x(), event.y())
            light_img = apply_light_effect(self.depth, light_position, self.radius, height_factor=self.height_factor, falloff=self.falloff, light_color=self.light_color)

            # Применяем multiply смешивание к img2 с учетом цвета света
            img2_multiplied_with_light = self.multiply_blend(self.img2, light_img, self.light_color)

            # Налагаем полученное изображение на исходное
            self.result_img = Image.alpha_composite(self.img, img2_multiplied_with_light)

            pixmap = self.convert_pil_to_pixmap(self.result_img)
            self.label.setPixmap(pixmap)


    def convert_pil_to_pixmap(self, img):
        data = img.tobytes("raw", "RGBA")
        qim = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qim)
        return pixmap


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LightApplication()
    sys.exit(app.exec_())