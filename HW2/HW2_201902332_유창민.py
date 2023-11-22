import streamlit as st
from PIL import Image
from PIL import ImageOps 
import matplotlib.pyplot as plt 
import numpy as np
from PIL import ImageFilter
import cv2

st.set_page_config(layout="wide", page_title='My DIP')
st.write("## Changmin's Space")
st.write(":cat:")
st.sidebar.write("## my sidebar")

def process_image_hw(upload):
    image = Image.open(upload)
    image_arr = np.array(image)
    col1.write("Input")
    col1.image(image)
    # output1 = np.random.randint(255, size=image_arr.shape).astype("uint8") # random output
    output1 = grayscale(image)
    col2.write("Grayscale")
    col2.image(output1)
    output2 = equalize(image)
    col3.write("Equalize")
    col3.image(output2)
    output3 = blur(image)
    col4.write("Blur")
    col4.image(output3)
    
def process_image_edge(upload):
    image = Image.open(upload)
    image_arr = np.array(image)
    output = laplacian(image_arr)
    col1.write("laplacian")
    col1.image(output)
    output1 = sobel(image_arr)
    col2.write("sobel")
    col2.image(output1)
    output2 = Scharr(image_arr)
    col3.write("Scharr")
    col3.image(output2)
    output3 = canny(image_arr)
    col4.write("canny")
    col4.image(output3)
    
def process_image_blur(upload):
    image = Image.open(upload)
    image_arr = np.array(image)
    output = sharpening(image_arr)
    col1.write("sharpening")
    col1.image(output)
    output1 = GaussianBlur(image_arr)
    col2.write("gaussianblur")
    col2.image(output1)
    output2 = boxfilter(image_arr)
    col3.write("boxfilter")
    col3.image(output2)
    output3 = medianBlur(image_arr)
    col4.write("medianBlur")
    col4.image(output3)   
    
def grayscale(image):
    return image.convert('L') # 흑백화

def equalize(image):
    image = ImageOps.grayscale(image)
    return ImageOps.equalize(image) # 히스토그램 평활화

def blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=5)) # 블러링

def canny(image):
    return cv2.Canny(image, 100, 200) # 물체 외곽선 추출

def Scharr(image):
    return cv2.Scharr(image, cv2.CV_8U, 1, 0) # 물체 외곽선 추출

def sobel(image):
    return cv2.Sobel(image, cv2.CV_8U, 1, 1, ksize=3) # 물체 외곽선 추출

def laplacian(image):
    return cv2.Laplacian(image, cv2.CV_8U) # 물체 외곽선 추출

def sharpening(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel) #컨볼루션 연산
    
def boxfilter(image):
    return cv2.boxFilter(image, -1, (3, 3)) # 블러링

def GaussianBlur(image):
    return cv2.GaussianBlur(image, (3, 3), 0) # 블러링

def medianBlur(image):
    return cv2.medianBlur(image, 5) # 블러링

col1, col2, col3, col4 = st.columns(4)

my_upload = st.sidebar.file_uploader("Upload", type=['png', 'jpg', 'jpeg'])

if my_upload is not None:
    process_image_hw(upload=my_upload)
    process_image_edge(upload=my_upload)
    process_image_blur(upload=my_upload)
else:
    process_image_hw("./test.jpg")
    process_image_edge("./test.jpg")
    process_image_blur("./test.jpg")