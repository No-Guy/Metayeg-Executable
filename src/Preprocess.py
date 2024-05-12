from PIL import Image 
import torchvision.transforms.functional as con 
import torchvision.transforms as transforms
import numpy as np
import cv2
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to the model's input shape
])
def Proc(imgpath):
    img = Image.open(imgpath)
    img = transform(img)
    #increased_contrast_img = con.adjust_contrast(img, 1.25) 
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(opencv_image)
    equalized_b = cv2.equalizeHist(b)
    equalized_g = cv2.equalizeHist(g)
    equalized_r = cv2.equalizeHist(r)
    tmp = cv2.merge([equalized_r, equalized_g, equalized_b])  
    pil_image = Image.fromarray(tmp)
    return pil_image