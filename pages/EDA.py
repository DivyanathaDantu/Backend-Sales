import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from statistics import mean
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from textwrap import wrap
from numerize import numerize

from MyUtils.HideStDefaults import hideNavBar
from MyUtils.Metrics import displayMetrics
from MyUtils.searchAndSelectFile import selectDataset

hideNavBar()

df = selectDataset()

#st.title("Sales Performance Dashboard")
#displayMetrics(df)

#taking the input from the user
df['Year'] = df['Year'].astype(str)
df[['Sales','Quantity','Rating']] = df[['Sales','Quantity','Rating']].apply(pd.to_numeric)
col1,col2,col3 = st.columns(3,gap="small")
with col1:
    input_year = st.multiselect(label="Choose Year", options=df["Year"].unique())
with col2:
    input_Quarter = st.multiselect(label="Choose Quarter", options=df["Quarter"].unique())
with col3:
    input_month = st.multiselect(label="Choose Month", options=df["Month"].unique())
if not input_year:
  input_year=df['Year'].unique()
if not input_Quarter:
  input_Quarter=df['Quarter'].unique()
if not input_month:
  input_month=df['Month'].unique()
filtered_df = df[ df['Year'].isin(input_year) &  df['Quarter'].isin(input_Quarter) & df['Month'].isin(input_month) ] 
#kpi df
df_kpi=filtered_df.groupby(['Year','Quarter','Month']).agg({'Sales':sum,'Quantity':mean,'Profit':sum,'Rating':mean}).reset_index()
df_kpi.rename(columns={'Sales':'Total Sales','Quantity':'Average Quantity per Order',},inplace=True)
sales_kpi=numerize.numerize(float(df_kpi['Total Sales'].sum()),2)
AQO_kpi=round( df_kpi['Average Quantity per Order'].mean() , 2 )
profit_kpi= numerize.numerize(float(df_kpi['Profit'].sum()),2)
Rating_kpi=round(df_kpi['Rating'].mean(), 2 )
col4, col5, col6, col7 = st.columns(4,gap="small")

with col4:
    st.metric(
        label="Sales",
        value=sales_kpi,
        
    )

with col5:
    st.metric(
        label="Average Order Quantity",
        value=AQO_kpi,
        
    )

with col6:
    st.metric(
        label="Profit",
        value=profit_kpi,
        
    )

with col7:
    st.metric(
        label="Rating",
        value=Rating_kpi,
        
    )
df_graph=filtered_df.groupby(['Year','Quarter','Month','Week number']).agg({'Sales':sum,'Quantity':mean,'Profit Margin':mean,'Rating':mean}).reset_index()
df_graph.rename(columns={'Sales':'Total Sales','Quantity':'Average Quantity per Order',},inplace=True)
#Weekly Sales

filtered_df_weekly_sales=filtered_df.groupby('Week number').agg({'Sales':sum}).reset_index()
fig_weekly_sales = px.area(filtered_df_weekly_sales,
                     x="Week number", y="Sales",
                     template="plotly_white", title="Weekly Total Sales",labels={"Week number":"Week","Sales":"Sales"},width=1000, height=400,color_discrete_sequence=['#B2182B'])
fig_weekly_sales.update_layout(title_x=0.5,xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
st.plotly_chart(fig_weekly_sales)
col8,padding, col9 = st.columns((2,2,1))

filtered_df_rating=filtered_df.groupby('Region').agg({'Rating':mean}).reset_index()
with col8:
    fig_filtered_df_rating = px.bar(filtered_df_rating,
                     x="Region", y="Rating",
                     template="plotly_white", title="Region wise Ratings",width=400, height=500,color_discrete_sequence=['#B2182B'])
    fig_filtered_df_rating .update_layout(title_x=0.5,xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    fig_filtered_df_rating.update_yaxes(range=[4.55, 4.7]) 
    st.plotly_chart(fig_filtered_df_rating)
    
#Region Sales
filtered_df_regionwise_sales=filtered_df.groupby('Region').agg({'Sales':sum}).reset_index()
with col9:
    fig_regionwise_sales = px.pie(filtered_df_regionwise_sales,
                     names="Region", values="Sales",
                     template="plotly_white", title="Region wise Sales Distribution",width=400, height=500,color_discrete_sequence=px.colors.sequential.RdBu)
    fig_regionwise_sales.update_layout(title_x=0.28)
    st.plotly_chart(fig_regionwise_sales)
    #rating
col10,padding, col11 = st.columns((2,1,2))
#Product Category
filtered_df_product=filtered_df.groupby('Product Category').agg({'Sales':sum}).reset_index()
with col8:
    fig_filtered_df_product = px.pie(filtered_df_product,
                     names='Product Category', values="Sales",
                     template="plotly_white", title="Product Category Distribution",width=400, height=500,color_discrete_sequence=px.colors.sequential.RdBu,hole=0.4)
    fig_filtered_df_product.update_layout(title_x=0.28)
    st.plotly_chart(fig_filtered_df_product)

#subcategory
filtered_df_subcategory=filtered_df.groupby('Sub Category').agg({'Quantity':sum}).reset_index()
with col9:
    fig_filtered_df_subcategory = px.line(filtered_df_subcategory,
                     x="Sub Category", y="Quantity",
                     template="plotly_white", title="Sub Category wise Orders",width=400, height=500)
    fig_filtered_df_subcategory .update_layout(title_x=0.28)
    fig_filtered_df_subcategory.update_traces(line_color='#B2182B')
    st.plotly_chart(fig_filtered_df_subcategory)