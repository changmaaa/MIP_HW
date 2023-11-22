import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## Spatial Filtering")
st.write(":dog:")
st.sidebar.write("## my sidebar")
plt.rcParams['font.size']=3

def my_conv2D(img, kernel):    
    SS = kernel.shape[0] # kernel size (for square kernel)
    ss = int((SS-1)/2)
    outimg = np.zeros(np.shape(img))
    for x in range(ss,np.size(img,0)-ss):
        for y in range(ss,np.size(img,1)-ss):
            Sxy = img[x-ss:x-ss+SS,y-ss:y-ss+SS]
            outimg[x,y] = np.sum(Sxy*kernel)
    return outimg

def my_median2D(image, kernel_size):
    SS = kernel_size # kernel size (square kernel)
    ss = int((SS-1)/2)
    r_median = np.zeros(np.shape(image))
    for x in range(ss,np.size(image,0)-ss):
        for y in range(ss,np.size(image,1)-ss):
            Sxy = image[x-ss:x-ss+SS,y-ss:y-ss+SS]
            r_median[x,y] = np.median(Sxy)
    return r_median

def process_image(upload):
    image = Image.open(upload)
    image = np.array(image.convert('L'))

    col1.write("Input, Box, Laplacian")
    fig1, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1,sharey=True)
    ax1.imshow(image,cmap='gray')
    ax1.axis('off')
    ax1.title.set_text('input')

    box = np.ones((7,7))*(1.0/49)
    output = my_conv2D(image, box)
    ax2.imshow(output,cmap='gray')
    ax2.axis('off')
    ax2.title.set_text('box')

    output = np.random.random(image.shape)
    ax3.imshow(output,cmap='gray')
    ax3.axis('off')
    ax3.title.set_text('1. laplacian')
    col1.pyplot(fig1)

    col2.write("Median, Min, Max")    
    fig2, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1,sharey=True)
    output = my_median2D(image, 7)
    ax1.imshow(output,cmap='gray')
    ax1.axis('off')
    ax1.title.set_text('Median')    
    
    output = np.random.random(image.shape)
    ax2.imshow(output,cmap='gray')
    ax2.axis('off')
    ax2.title.set_text('2. Min')

    output = np.random.random(image.shape)
    ax3.imshow(output,cmap='gray')
    ax3.axis('off')
    ax3.title.set_text('3. Max')    
    col2.pyplot(fig2)


    col3.write("Adaptive Filters")
    fig3, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1,sharey=True)
    output = np.random.random(image.shape)
    ax1.imshow(output,cmap='gray')
    ax1.axis('off')
    ax1.title.set_text('4. adaptive local noise reduction')    
    
    output = np.random.random(image.shape)
    ax2.imshow(output,cmap='gray')
    ax2.axis('off')
    ax2.title.set_text('5. adaptive median')

    output = np.random.random(image.shape)
    ax3.imshow(output,cmap='gray')
    ax3.axis('off')
    ax3.title.set_text('random noise')

    col3.pyplot(fig3)



col1, col2, col3 = st.columns(3)
my_upload = st.sidebar.file_uploader("Upload", type=["png","jpg","jpeg","tif"])

if my_upload is not None:
    process_image(upload=my_upload)