import torchvision.transforms.functional as TF
from torchvision import transforms

def tta_horizontal_flip(img):
    return TF.hflip(img)

def tta_vertical_flip(img):
    return TF.vflip(img)

def tta_rotation_90(img):
    return TF.rotate(img, 90)
def tta_rotation_180(img):
    """旋转180度"""
    return TF.rotate(img, 180)

def tta_rotation_270(img):
    """旋转270度"""
    return TF.rotate(img, 270)

def tta_color_jitter(img):
    """色彩抖动"""
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    return color_jitter(img)


