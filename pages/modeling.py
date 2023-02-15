import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from MyUtils.HideStDefaults import hideNavBar
from MyUtils.searchAndSelectFile import selectDataset

hideNavBar()


df = selectDataset()
# Ask the customer to choose Independant variables from the list of "df" column names.The chosen data should be fed
# into a list "a"
a = ["Parameter 1", "Parameter 2", "Parameter 3", "Parameter 4", "Parameter 5"]
# Ask the customer to choose Dependant variables from the list of "df" column names.The chosen data should be fed
# into a list "b"
b = ["Total sales"]

# Metric 1 - To be placed at bottom  centre


x = df[a]
y = df[b].values.ravel()

df_model = pd.concat([df[a], df[b]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Metric 2  -  Top be placed at top left
importance = model.feature_importances_
c = list(x_test.columns)
d = importance.tolist()
fi = [c, d]
df_fi = pd.DataFrame(fi).transpose()
df_fi.columns = ['Feature', 'Importance']
fig4 = px.bar(df_fi,
              x="Importance", y="Feature",
              template="plotly_white", labels={"Feature": "Feature", "Importance": "Importance"}, width=1500,
              height=400,
              color_discrete_sequence=['#00CC96'], orientation='h')
# st.bar_chart(data=fig4,  width=0, height=0, use_container_width=True)
# st.markdown("<h1 style='text-align: center; color: Black;'>Feature Importance</h1>", unsafe_allow_html=True)
st.plotly_chart(fig4)

# Metric 3 -  Top be placed at top right


correlation_matrix = df_model.corr()

fig7 = plt.figure(figsize=(20, 15))  # For Size increase
sns.heatmap(correlation_matrix, annot=True, linewidths=.5, cmap="Blues", vmax=0.5, annot_kws={"size": 24})
# st.markdown("<h1 style='text-align: center; color: Black;'>Correlation Matrix</h1>", unsafe_allow_html=True)
st.write(fig7)
