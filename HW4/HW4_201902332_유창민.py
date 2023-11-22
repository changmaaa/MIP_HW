import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal as signal
import scipy.ndimage as nd

st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## Connected Components")
st.write(":dog:")
st.sidebar.write("## my sidebar")
plt.rcParams['font.size']=3

def process_image(upload):
    I = Image.open(upload)
    f = np.array(I)
    f = f.astype(np.float32) / 255
    r = f[:, :, 0]
    g = f[:, :, 1]
    b = f[:, :, 2]
    res = (r+g)/2
    temp = res - b*0.6
    f2 = temp > 0.45
    f3 = nd.binary_opening(f2, np.ones((3,3)))
    f3 = nd.binary_closing(f3, np.ones((3,3)))
    f3 = nd.binary_closing(f3, np.ones((3,3)))
    f3 = nd.binary_opening(f3, np.ones((3,3)))
    l, m = nd.label(f3)
    col1.write("Original image")
    fig1, (ax1) = plt.subplots(nrows=1, ncols=1,sharey=True)
    ax1.imshow(f)
    ax1.axis('off')
    col1.pyplot(fig1)
    col2.write("Binary mask")
    fig2, (ax1) = plt.subplots(nrows=1, ncols=1,sharey=True)
    ax1.imshow(f2, cmap='gray')
    ax1.axis('off')
    col2.pyplot(fig2)
    col3.write("Connected components")
    fig3, (ax1) = plt.subplots(nrows=1, ncols=1,sharey=True)
    ax1.imshow(f3,interpolation='nearest',cmap='tab20c')
    ax1.axis('off')
    col3.pyplot(fig3)
    cc = np.zeros((m-1,2))
    th_val = 250
    cnt = 0
    for i in range(1,m):
        cc[i-1,0] = i
        cc[i-1,1] = np.sum(l==i)
        if cc[i-1,1] < th_val:
            cnt += 1
    st.write("Total number of connected components(bigger than 20pixels):", m-1-cnt)
    ind_max = np.argmax(cc[:, 1])
    st.write("Number of pixels in the largest connected component:", int(cc[ind_max, 1]))
    

col1, col2, col3 = st.columns(3)
my_upload = st.sidebar.file_uploader("Upload", type=["png","jpg","jpeg","tif"])

if my_upload is not None:
    process_image(upload=my_upload)