from transformers import pipeline
from PIL import Image
import cv2
import numpy as np

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

def OcclusionMask(image):
    image = Image.fromarray(image.transpose(1, 2, 0))
    depth = pipe(image)["depth"]
    depth = np.array(depth)
    mask = cv2.Sobel(depth, cv2.CV_64F, 1, 1, ksize=3)
    mask = cv2.convertScaleAbs(mask)
    mask= cv2.threshold(mask, 2, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.dilate(mask, None, iterations=4)[None] / 255.0
    mask = 1 - mask
    return mask.astype(np.float32)

if __name__ == '__main__':
    mask = OcclusionMask("test.jpg")
    mask.save("mask.jpg")
