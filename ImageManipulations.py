from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ImageManipulations:
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.manipulated = self.image.copy()

    # 1. Create a negative of the image
    def negative(self):
        self.manipulated = ImageOps.invert(self.image.convert('RGB'))

    # 2. Add a color tint to the image
    def add_tint(self, color=(255, 0, 0), intensity=0.5):
        overlay = Image.new('RGB', self.image.size, color)
        self.manipulated = Image.blend(self.image, overlay, intensity)

    # 3. Crop an image (left, top, right, bottom)
    def crop(self, crop_box):
        self.manipulated = self.image.crop(crop_box)

    # 4. Adjust brightness
    def adjust_brightness(self, factor=1.5):
        enhancer = ImageEnhance.Brightness(self.image)
        self.manipulated = enhancer.enhance(factor)

    # 5. Adjust contrast
    def adjust_contrast(self, factor=1.5):
        enhancer = ImageEnhance.Contrast(self.image)
        self.manipulated = enhancer.enhance(factor)

    # 6. Convert an image to grayscale
    def to_grayscale(self):
        # Convert the PIL Image to a NumPy array
        image_array = np.array(self.image)
        # Convert the NumPy array to grayscale using cv2.cvtColor
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        self.manipulated = Image.fromarray(gray_image)

    # 7. Add a shape or picture on top of the image
    def add_shape(self, shape='circle', position=(100, 100), size=50, color=(255, 0, 0)):
        draw = ImageDraw.Draw(self.image)
        if shape == 'circle':
            draw.ellipse((position[0] - size, position[1] - size, position[0] + size, position[1] + size), fill=color)
        elif shape == 'rectangle':
            draw.rectangle([position, (position[0] + size, position[1] + size)], fill=color)

        self.manipulated = self.image

    # 8. Add text to the image
    def add_text(self, text, position=(10, 10), font=None, color=(255, 255, 255)):
        draw = ImageDraw.Draw(self.image)
        if font is None:
            font = ImageFont.load_default()
        draw.text(position, text, fill=color, font=font)
        self.manipulated = self.image

    # 9. Histogram equalization
    def histogram_equalization(self):
        image_array = np.array(self.image.convert('L'))
        image_eq = Image.fromarray(cv2.equalizeHist(image_array))
        self.manipulated = image_eq

    # 10. Scale the image (up or down scaling)
    def scale(self, width=None, height=None):
        self.manipulated = self.image.resize((width, height))

    # 11. Translate the image
    def translate(self, x, y):
        self.manipulated = self.image.transform(self.image.size, Image.AFFINE, (1, 0, x, 0, 1, y))

    # 12. Rotate the image
    def rotate(self, angle):
        self.manipulated = self.image.rotate(angle)

    # 13. Blur the image
    def blur(self):
        self.manipulated = self.image.filter(ImageFilter.BLUR)

    # 14. Sharpen the image
    def sharpen(self):
        enhancer = ImageEnhance.Sharpness(self.image)
        self.manipulated = enhancer.enhance(2.0)

    # 15. Edge detection
    def edge_detection(self):

        self.manipulated = self.image.filter(ImageFilter.FIND_EDGES)

    def show_manipulated(self):
        plt.imshow(self.manipulated)
        plt.axis('off')
        plt.show()

    def show_source(self):
        plt.imshow(self.image)
        plt.axis('off')
        plt.show()

    def save(self, output_path):
        self.manipulated.save(output_path)