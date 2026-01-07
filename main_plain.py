import cv2
import easyocr
import numpy as np
import re
import matplotlib.pyplot as plt

#image path
image_path = 'Screenshot From 2025-10-27 14-20-04.png'

#load image
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Image not found")

#preprocess image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = gray.astype('uint8')

blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

#OCR
reader = easyocr.Reader(['en'], gpu = False)
result = reader.readtext(thresh)

#print results
print("\n OCR OUTPUT (confidence > 0.5):\n")

for box, text, conf in result:
    if conf > 0.5:
        print (text, " | confidence:", round(conf, 2))
    
#Show image
plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
    