import cv2
import PIL
from PIL import Image
import numpy as np


def image_processing(image: Image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # ############# Contrast, Saturation and Sharpness ############# #
    brightness = 5
    contrast = 1.1

    # Apply contrast
    image_contrast = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)

    # Convert image to HSV
    image_hsv = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2HSV)

    # Apply saturation
    saturation = 1.1  # Saturation scale value
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * saturation, 0, 255)

    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # Sharpness kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # Apply sharpness filter
    image_sharp = cv2.filter2D(image_rgb, -1, kernel)
    image_sharp = cv2.cvtColor(image_sharp, cv2.COLOR_RGB2BGR)
    return PIL.Image.fromarray(image_sharp)
