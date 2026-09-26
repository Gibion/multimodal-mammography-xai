import numpy as np
import cv2
from torchvision import transforms

class Preprocess:
    def __call__(self, img):
        # img: HxWx3 (RGB) or HxW (grayscale)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.equalizeHist(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        return img

def get_preprocessing_transforms():
    return transforms.Compose([
        Preprocess(),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])
