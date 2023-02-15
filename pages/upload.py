import os

import pandas as pd
import streamlit as st
from MyUtils.HideStDefaults import hideNavBar

hideNavBar()

@st.cache
def load_data(file):
    if file:
        return pd.read_csv(file)
    else:
        return None
st.title("Search Your Dataset")
file = st.file_uploader("Upload Your Dataset", type=["csv"])

if file:
    df = pd.read_csv(file, index_col=None)
    df.to_csv('dataset.csv', index=None)

     # Sidebar - Collects user input features into dataframe
    st.dataframe(df)
