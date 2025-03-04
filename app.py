import sys
import os
import glob
import re
import numpy as np
import streamlit as st

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from haroun import Data, Model, ConvPool
from haroun.augmentation import augmentation
from haroun.losses import rmse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



st.title("Real or Fake Image Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])




class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.input_norm = torch.nn.BatchNorm2d(3, affine=False)
        self.layer1 = ConvPool(in_features=3, out_features=8)
        self.layer2 = ConvPool(in_features=8, out_features=16)
        self.layer3 = ConvPool(in_features=16, out_features=32)
        self.layer4 = ConvPool(in_features=32, out_features=64)
        self.layer5 = ConvPool(in_features=64, out_features=128)
        self.layer6 = ConvPool(in_features=128, out_features=256)
        
        

        self.net = torch.nn.Sequential(self.layer1, self.layer2, self.layer3, 
                                       self.layer4, self.layer5, self.layer6)
            
        
        self.fc1 = torch.nn.Linear(in_features=256, out_features=128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        
        self.fc2 = torch.nn.Linear(in_features=128, out_features=32)
        self.bn2 = torch.nn.BatchNorm1d(32)

        self.fc3 = torch.nn.Linear(in_features=32, out_features=8)
        self.bn3 = torch.nn.BatchNorm1d(8)

        self.fc4 = torch.nn.Linear(in_features=8, out_features=2)


        self.lin = torch.nn.Sequential(self.fc1, self.bn1, self.fc2, self.bn2,
                                       self.fc3, self.bn3, self.fc4)  


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.input_norm(X)
        X = self.net(X)
        X = X.reshape(X.size(0), -1)
        X = self.lin(X)
        X = torch.nn.functional.elu(X, alpha=1.0, inplace=False)
        return X

PATH = 'best.pt'

net = Network()
net.load_state_dict(torch.load(PATH, map_location=device))
net.eval() 

CTS = PATH

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    image = image.resize((64, 64))
    image = ToTensor()(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = net(image)

    if output[0][0] < output[0][1]:
        st.write("Based on the Uploaded Image, it is a fake image.")
    else:
        st.write("Based on the Uploaded Image, it is a real image.")


menu = st.sidebar.selectbox("Navigation", ["Home"])
if menu == "Home":
    st.write("Welcome to the Real or Fake Image Detection App.")

