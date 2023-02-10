import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import statsmodels.api as sm

# Tasks to solve
# Input data for fit and predict
# Train and test split
# from fedot.core.data.data import train_test_data_setup
#st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
section[class="css-163ttbj e1fqkh3o11"] {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.title("Upload Your Dataset")
file = st.file_uploader("Upload Your Dataset", type=["csv"])
   
if file: 
    df = pd.read_csv(file, index_col=None)
    df.to_csv('dataset.csv', index=None)
        
     # Sidebar - Collects user input features into dataframe
    st.dataframe(df)
      
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)







 