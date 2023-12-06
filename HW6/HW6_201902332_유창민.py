import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tf

st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## Handwritten Digit Recognition")
st.sidebar.write("## my sidebar")
plt.rcParams['font.size'] = 12


def process_image(upload):
    # image = Image.open(upload)
    # image = np.array(image.convert('L'))
    # input = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # input = input / torch.max(input)
    # input = tf.Resize((34, 34))(input)
    # input = tf.CenterCrop(28)(input)
    transform = tf.Compose(
    [
        tf.ToTensor(),
        tf.Grayscale(),
        tf.Lambda(lambda x: (x > 0.7).float()),
        tf.Resize(size=(32, 32)),
        tf.CenterCrop(28),
    ])
    image = Image.open(upload)
    image = transform(image)
    input = image.unsqueeze(0)

    net = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=16*5*5, out_features=120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 84),
        torch.nn.ReLU(),
        torch.nn.Linear(84, 10)
    )
    net.load_state_dict(torch.load(f'/Users/yuchangmin/3_2/digital_image/HW/HW6/my_best_model_90.000.pth', map_location=torch.device('cpu')))

    # Extracting intermediate layer output
    activations = []

    def hook(module, input, output):
        activations.append(output)

    net[0].register_forward_hook(hook)

    net.eval()
    with torch.no_grad():
        out = net(input)
    
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharey=True)
    ax1.imshow(input[0, 0], cmap='gray')
    ax1.axis('off')
    ax1.title.set_text('input')
    col1.pyplot(fig1)
    
    if len(activations) > 0:
        activation = activations[0].squeeze().detach().cpu()
        num_activations = activation.shape[0]
        rows = 2
        cols = 3
        if num_activations > rows * cols:
            st.warning(f"Too many feature maps to display. Showing first {rows * cols} feature maps.")
            num_activations = rows * cols
        fig2, ax2 = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 6))
        for idx in range(num_activations):
            row = idx // cols
            col = idx % cols
            ax2[row, col].imshow(activation[idx], cmap='gray')
            ax2[row, col].axis('off')
        col2.write("First Convolutional layer output")
        col2.pyplot(fig2)

    top_classes = torch.argsort(out, descending=True)
    probabilities = torch.softmax(out, dim=1)[0]
    count = 0
    for i, prob in enumerate(probabilities):
        count += 1
        class_name = top_classes[0][i].item()
        col3.write(f"{i+1}nd prediction: Number {class_name}, Probability: {prob.item()}")
        if count == 5:
            break
    st.write(f"answer: {torch.argmax(out.softmax(1), axis=1).detach().numpy()[0]}")

col1, col2, col3 = st.columns(3)
my_upload = st.sidebar.file_uploader("Upload", type=["png", "jpg", "jpeg", "tif"])

if my_upload is not None:
    process_image(upload=my_upload)
