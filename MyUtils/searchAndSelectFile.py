import pandas as pd
import streamlit as st

from MyUtils.Firebase import get_firestore_files


def selectDataset():
    search = st.text_input("Search Your Dataset")

    files = get_firestore_files(search)
    search_files = {}
    for doc in files:
        # print(f'{doc.id} => {doc.to_dict()}')
        search_files[doc.to_dict()['file_name']] = doc.to_dict()['file_url']
    file = st.selectbox('Select Your Dataset',
                        search_files.keys())

    if file:
        st.write('You selected `%s`' % file)
        df = pd.read_csv(search_files[file], index_col=None, encoding_errors='replace')
        st.dataframe(df.head(5))
        return df
