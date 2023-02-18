import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from MyUtils.HideStDefaults import hideNavBar
from MyUtils.Metrics import displayMetrics
from MyUtils.searchAndSelectFile import selectDataset
#dsddsds
hideNavBar()

df = selectDataset()

st.title("Sales Performance Dashboard")
displayMetrics(df)
y = [2015, 2016]
q = [1, 2]
m = ["January", "February"]
# st.header("Sales Performance Dashboard")

# Creating of filtered database
df_q = df.query("Year in @y & Quarter in @q & Month in @m")

df_c1 = df_q.groupby('Week').agg({'Total sales': sum}).reset_index()
fig1 = px.area(df_c1, x="Week", y="Total sales", template="plotly_white", title="Weekly Sales",
               labels={"Week": "Week", "Total sales": "Sales"}, width=1000, height=400)
plt.rcParams["figure.figsize"] = (7, 3)
st.plotly_chart(fig1)

df_c2 = df_q.groupby('Sales Manager').agg({'Total sales': sum}).reset_index()
fig2 = px.bar(df_c2,
              x="Sales Manager", y="Total sales",
              template="plotly_white", title="Top Sales Representatives",
              labels={"Sales Manager": "Sales Representatives", "Total  sales": "Sales"}, width=600, height=400,
              color_discrete_sequence=['#00CC96'])

df_c3 = df_q.iloc[:, 5:10].sum(axis=0).reset_index()
df_c3.rename(columns={df_c3.columns[0]: "Product", df_c3.columns[1]: "Sales"}, inplace=True)
fig3 = px.pie(df_c3,
              values='Sales', names='Product',
              template="plotly_white", title="Product Sales Distribution", width=500, height=500,
              color_discrete_sequence=['#EF553B'])

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig3, use_column_width=True)

with col2:
    st.plotly_chart(fig2)
