import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def AdaptiveLocalNoiseReduction(img, kernel_size):
    SS = kernel_size # kernel size (square kernel)
    ss = int((SS-1)/2)
    std_eta = 16
    outimg = np.zeros(np.shape(img))
    for x in range(ss,np.size(img,0)-ss):
        for y in range(ss,np.size(img,1)-ss):
            Sxy = img[x-ss:x-ss+SS,y-ss:y-ss+SS]
            std_Sxy = np.std(Sxy)
            if std_eta<=std_Sxy:
                outimg[x,y]=img[x,y]-((std_eta*std_eta)/(std_Sxy*std_Sxy))*(img[x,y]-np.mean(Sxy))
            else:
                outimg[x,y]=np.mean(Sxy)
    return outimg

def AdaptiveMedianFilter(img):
    f = img
    SS_vals = np.array([3, 5, 7])
    r_amedian = np.zeros(np.shape(f))
    r_median = np.zeros(np.shape(f))
    r_case = np.zeros(np.shape(f))
    
    ss_max = int((np.max(SS_vals)-1)/2)
    for x in range(ss_max,np.size(f,0)-ss_max):
        for y in range(ss_max,np.size(f,1)-ss_max):
            flag_go = 1
            cnt = 0
            z_xy = f[x,y]
            for SS in SS_vals:
                ss = int((SS-1)/2)
                Sxy = f[x-ss:x-ss+SS,y-ss:y-ss+SS]
                z_med = np.median(Sxy)
                z_min = np.min(Sxy)
                z_max = np.max(Sxy)

                if cnt==0:
                    r_median[x,y] = z_med

                if z_min<z_med and z_med<z_max:
                    if z_min<z_xy and z_xy<z_max:
                        r_amedian[x,y] = z_xy
                        flag_go = 0
                    else:
                        r_amedian[x,y] = z_med
                        flag_go = 0
                else:
                    if SS == np.max(SS_vals):
                        r_amedian[x,y] = z_med
                        flag_go = 0

                if flag_go==0:
                    r_case[x,y] = cnt
                    break
                cnt = cnt+1
    return r_amedian

def AdaptiveMedianFilter_CM(img, kernel_size):
    SS = kernel_size  # 초기 커널 크기
    ss = int((SS - 1) / 2)
    outimg = np.zeros(np.shape(img))
    zmin = np.zeros(np.shape(img))
    zmax = np.zeros(np.shape(img))
    zmed = np.zeros(np.shape(img))
    for x in range(ss, np.size(img, 0) - ss):
        for y in range(ss, np.size(img, 1) - ss):
            # Ensure the selected region doesn't exceed image dimensions
            Sxy = img[max(0, x - ss) : min(np.size(img, 0), x - ss + SS), 
                      max(0, y - ss) : min(np.size(img, 1), y - ss + SS)]
            
            # Check if Sxy is non-empty
            if Sxy.size > 0:
                zmin[x, y] = np.min(Sxy)
                zmax[x, y] = np.max(Sxy)
                zmed[x, y] = np.median(Sxy)
                zxy = img[x, y]
                if zmed[x, y] - zmin[x, y] > 0 and zmax[x, y] - zmed[x, y] > 0:
                    if zxy - zmin[x, y] > 0 and zmax[x, y] - zxy > 0:
                        outimg[x, y] = zxy
                    else:
                        outimg[x, y] = zmed[x, y]
                else:
                    SS = SS+2
                    ss = int((SS-1)/2)
                    Sxy = img[max(0, x - ss) : min(np.size(img, 0), x - ss + SS), 
                              max(0, y - ss) : min(np.size(img, 1), y - ss + SS)]
                    zmin[x,y] = np.min(Sxy)
                    zmax[x,y] = np.max(Sxy)
                    zmed[x,y] = np.median(Sxy)
                    zxy = img[x,y]
                    if zmed[x,y]-zmin[x,y]>0 and zmax[x,y]-zmed[x,y]>0:
                        if zxy-zmin[x,y]>0 and zmax[x,y]-zxy>0:
                            outimg[x,y] = zxy
                        else:
                            outimg[x,y] = zmed[x,y]
                    else:
                        outimg[x,y] = zmed[x,y]
    return outimg
# 이건 내가 짠 거
#kernel size 3,5,7로 고정하지 않고 늘려가면서 하는 방식

def min2D(image, kernel_size):
    SS = kernel_size # kernel size (square kernel)
    ss = int((SS-1)/2)
    r_min = np.zeros(np.shape(image))
    for x in range(ss,np.size(image,0)-ss):
        for y in range(ss,np.size(image,1)-ss):
            Sxy = image[x-ss:x-ss+SS,y-ss:y-ss+SS]
            r_min[x,y] = np.min(Sxy)
    return r_min

def max2D(image, kernel_size):
    SS = kernel_size # kernel size (square kernel)
    ss = int((SS-1)/2)
    r_max = np.zeros(np.shape(image))
    for x in range(ss,np.size(image,0)-ss):
        for y in range(ss,np.size(image,1)-ss):
            Sxy = image[x-ss:x-ss+SS,y-ss:y-ss+SS]
            r_max[x,y] = np.max(Sxy)
    return r_max

def my_median2D(image, kernel_size):
    SS = kernel_size # kernel size (square kernel)
    ss = int((SS-1)/2)
    r_median = np.zeros(np.shape(image))
    for x in range(ss,np.size(image,0)-ss):
        for y in range(ss,np.size(image,1)-ss):
            Sxy = image[x-ss:x-ss+SS,y-ss:y-ss+SS]
            r_median[x,y] = np.median(Sxy)
    return r_median

def laplacian2D(image):
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    return my_conv2D(image, kernel)

def process_image(upload):
    image = Image.open(upload)
    image = np.array(image.convert('L'))

    col1.write("Input, Min, Max")
    fig1, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1,sharey=True)
    ax1.imshow(image,cmap='gray')
    ax1.axis('off')
    ax1.title.set_text('input')

    box = np.ones((7,7))*(1.0/49)
    output = min2D(image, 3)
    ax2.imshow(output,cmap='gray')
    ax2.axis('off')
    ax2.title.set_text('min')

    output = max2D(image, 3)
    ax3.imshow(output,cmap='gray')
    ax3.axis('off')
    ax3.title.set_text('max')
    col1.pyplot(fig1)

    col2.write("Laplacian2D, Adaptive Local Noise Reduction, Adaptive Median")    
    fig2, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1,sharey=True)
    output = laplacian2D(image)
    ax1.imshow(output,cmap='gray')
    ax1.axis('off')
    ax1.title.set_text('LaPlacian2D')    
    
    output = AdaptiveLocalNoiseReduction(image, 3)
    ax2.imshow(output,cmap='gray')
    ax2.axis('off')
    ax2.title.set_text('Adaptive Local Noise Reduction')

    output = AdaptiveMedianFilter(image)
    ax3.imshow(output,cmap='gray')
    ax3.axis('off')
    ax3.title.set_text('Adaptive Median')    
    col2.pyplot(fig2)


    # col3.write("Adaptive Filters")
    # fig3, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1,sharey=True)
    # output = np.random.random(image.shape)
    # ax1.imshow(output,cmap='gray')
    # ax1.axis('off')
    # ax1.title.set_text('4. adaptive local noise reduction')    
    
    # output = np.random.random(image.shape)
    # ax2.imshow(output,cmap='gray')
    # ax2.axis('off')
    # ax2.title.set_text('5. adaptive median')

    # output = np.random.random(image.shape)
    # ax3.imshow(output,cmap='gray')
    # ax3.axis('off')
    # ax3.title.set_text('random noise')

    # col3.pyplot(fig3)



col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload", type=["png","jpg","jpeg","tif"])

if my_upload is not None:
    process_image(upload=my_upload)