import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from MyUtils.HideStDefaults import hideNavBar
from MyUtils.searchAndSelectFile import selectDataset

hideNavBar()

df = pd.read_csv("trainingdata.csv")
st.title("Upload Your test Dataset")
df_testdata = selectDataset()

chosen_main_model = st.selectbox('Select Model Category',
                                 ["Choose an option", "Classification", "Regression", "Time Series"])

if chosen_main_model == "Classification":
    chosen_target_x = st.multiselect(label="Choose Independant  variable", options=df.columns)
    # d= st.selectbox(label = "Choose Dependant  variable", options = df["Total sales"])
    chosen_target_y = st.selectbox(label="Choose Dependant  variable",
                                   options=(df.columns).insert(0, "Choose an option"))
    if (chosen_target_y != "Choose an option") and chosen_target_x:
        chosen_class_model = st.selectbox('Select the Model',
                                          ["Choose an option", "Random Forest Classification", "KNN", "XGBoost",
                                           "Decision Tree"])
        if chosen_class_model == "Random Forest Classification":
            st.write("rendom forest regression")
        if chosen_class_model == "KNN":
            st.write("knn")
        if chosen_class_model == "XGBoost":
            st.write("XGBoost")
        if chosen_class_model == "Decision Tree":
            st.write("decision tree")

if chosen_main_model == "Regression":
    chosen_target_x = st.multiselect(label="Choose Independant  variable", options=df.columns)
    # d= st.selectbox(label = "Choose Dependant  variable", options = df["Total sales"])
    chosen_target_y = st.selectbox(label="Choose Dependant  variable", options=df.columns)
    chosen_class_model = st.selectbox('Select the Model',
                                      ["Linear Regression", "Logistic Regression", "Random Forest Regressor",
                                       "XGBoost Regression", "KNN regression"])
    if chosen_class_model == "Linear Regression":
        st.write("Linear Regression")
    if chosen_class_model == "Logistic Regression":
        st.write("Logistic Regression")
    if chosen_class_model == "Random Forest Regressor":
        st.write("Random Forest Regressor")
        x = df[chosen_target_x]
        y = df[chosen_target_y].values.ravel()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        model_train = RandomForestRegressor()
        model_train.fit(x_train, y_train)
        y_pred = model_train.predict(x_test)
        r2 = r2_score(y_test, y_pred).round(4)
        mae = mean_absolute_error(y_test, y_pred).round(4)
        mape = (mean_absolute_percentage_error(y_test, y_pred).round(4)) * 100
        mse = mean_squared_error(y_test, y_pred).round(4)
        rmse = mean_squared_error(y_test, y_pred, squared=False).round(4)
        max = max_error(y_test, y_pred).round(4)

        m_values = [r2, mae, mape, mse, rmse, max]
        m_title = ["R2", "Mean Absolute Error", "Mean Absolute Percentage Error", "Mean Square Error",
                   "Root Mean Square Error", "Maximum Error"]
        m = [m_title, m_values]
        df_m = pd.DataFrame(m).transpose()
        df_m.columns = ["Metric", "Value"]

        # Display the below table at the bottom of the page
        st.subheader("Model Performance")
        st.write(df_m)

        # Run the prediction and output the data

        model = RandomForestRegressor()
        x_4train = df[chosen_target_x]
        y_4train = df[chosen_target_y].values.ravel()
        model.fit(x_4train, y_4train)
        x_4pred = df_testdata[chosen_target_x]
        y_4pred = model.predict(x_4pred)
        df_y4pred = pd.DataFrame(y_4pred, columns=[chosen_target_y])
        df_download = pd.concat([df_testdata, df_y4pred], axis=1)

        st.subheader("Model Results")
        st.write(df_download)


        @st.cache
        def convert_df(df1):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df1.to_csv().encode('utf-8')


        csv = convert_df(df_download)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='large_df.csv',
            mime='text/csv',
        )

    if chosen_class_model == "XGBoost Regression":
        st.write("XGBoost Regression")
    if chosen_class_model == "KNN regression":
        st.write("KNN regression")

if chosen_main_model == "Time Series":
    chosen_target_x = st.multiselect(label="Choose Independant  variable", options=df.columns)
    # d= st.selectbox(label = "Choose Dependant  variable", options = df["Total sales"])
    chosen_target_y = st.selectbox(label="Choose Dependant  variable", options=df.columns)
    chosen_class_model = st.selectbox('Select the Model', ["SARIMAX", "Prophet", "ARIMA"])
    if chosen_class_model == "SARIMAX":
        st.write("SARIMAX")
    if chosen_class_model == "Prophet":
        st.write("Prophet")
    if chosen_class_model == "ARIMA":
        st.write("ARIMA")

############################################################################
