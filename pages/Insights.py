import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from MyUtils.HideStDefaults import hideNavBar
from MyUtils.searchAndSelectFile import selectDataset_with_msg
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular
#from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from MyUtils.HideStDefaults import hideNavBar
from MyUtils.Metrics import displayMetrics
from MyUtils.searchAndSelectFile import selectDataset

hideNavBar()


df_train = selectDataset_with_msg("Select your Training dataset")
df_test = selectDataset_with_msg("Select your Test dataset")

# df_train=pd.read_csv('training_data_insights.csv')
# df_test=pd.read_csv('test_data_insights.csv')


chosen_target_X = st.multiselect(label="Choose Independant  variable", options=df_train.columns)
chosen_target_Y = st.selectbox(label="Choose Dependant  variable",
                                   options=(df_train.columns).insert(0, "Choose an option"))

if chosen_target_Y != 'Choose an option':
    #Training the model
    #Define the ensemble model
    model_churn=VotingClassifier(estimators=[('LR', LogisticRegression(max_iter=1000)), ('RF', RandomForestClassifier()), ('GB', GradientBoostingClassifier())],
                            voting='soft')
    X_train=df_train[chosen_target_X]
    Y_train=df_train[chosen_target_Y]
    model_churn.fit(X_train.values,Y_train.values.ravel())

   #Predicting churn probability
    df_churn=df_test.copy()
    df_churn['churn_probability']=np.nan
    churn_probability_predictions=model_churn.predict_proba((df_churn[chosen_target_X]).values)
    df_churn['churn_probability']=churn_probability_predictions
    
    #Tile  1 - Total  active customers
    total_active_customers=len(df_test.index)
    #Tile 2 - Customers at the risk of churn
    df_risky_customers=(df_churn[df_churn['churn_probability']>0.50])
    total_risky_customers=len(df_risky_customers.index)
    #Tile 3  Overall loss
    overall_annual_loss=(df_risky_customers['Monthly Charges'].sum())*12
    col4, col5, col6, = st.columns(3,gap="small")

    with col4:
        st.metric(
        label="Total Active Customers",
        value=total_active_customers,
        
    )

    with col5:
        st.metric(
        label="Customers at the risk of Churn",
        value=total_risky_customers,
        
    )

    with col6:
        st.metric(
        label="Potential Annual loss due to Customer Churn",
        value=overall_annual_loss,
        
    )
    
    #Graph 1 -  Overall feature importance
    col21, col22 = st.columns(2,gap="small")
    with col21:

        fi = permutation_importance(model_churn,X_train.values,Y_train.values.ravel(), n_repeats=1,random_state=0)

        r_list=list(fi.importances_mean)
        X_list=list(X_train.columns)


        fi=[X_list,r_list]
        df_fi=pd.DataFrame (fi).transpose()
        df_fi.columns = ['Feature', 'Importance']
        df_fi=(df_fi.sort_values(by='Importance',ascending=True).reset_index(drop=True)).copy()

        fig_fi = px.bar(df_fi,
                            x="Importance", y="Feature",
                            template="plotly_white", title="Feature Importance",labels={"Feature":"Feature","Importance":"Importance"},width=300, height=400,
                            color_discrete_sequence=['#B2182B'], orientation='h')
        fig_fi.update_layout( xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),title_x=0.5)
        st.plotly_chart(fig_fi)

    #with col22:
            #Top & bottom 3 influencing parameters 
        #first_influencing_parameter=(df_fi.iloc[[0],[0]].values)[0][0]
        #second_influencing_parameter=(df_fi.iloc[[1],[0]].values)[0][0]
        #third_influencing_parameter=(df_fi.iloc[[2],[0]].values)[0][0]

        #least_influencing_parameter=(df_fi.iloc[[-1],[0]].values)[0][0]
        #secondleast_influencing_parameter=(df_fi.iloc[[-2],[0]].values)[0][0]
        #thirdleast_influencing_parameter=(df_fi.iloc[[-3],[0]].values)[0][0]    
                
        

    col25, col26 = st.columns(2,gap="small")
    with col25:
        col23,col24=st.columns(2,gap='small')
        #st.text("")
        #st.text("")
        #st.text("")
        #st.text("")
        #st.text("")
        with col23:
            query_variable=st.selectbox(label="Choose Query Variable", options=df_train.columns)
        with col24:
            query_variable_value=st.selectbox(label="Choose Query Value", options=df_train[query_variable].unique())
        #Querying the dataset
        condition = (df_train[query_variable] == query_variable_value) 
        df_queried=(df_train[condition]).copy().reset_index(drop=True)

        #loading the input data into the variables
        X_q=df_queried[chosen_target_X]
        Y_q=df_queried[chosen_target_Y]
        model_churn.fit(X_q.values,Y_q.values.ravel())

        fi_q = permutation_importance(model_churn,X_q.values,Y_q.values.ravel(), n_repeats=1,random_state=0)

        r_list_q=list(fi_q.importances_mean)
        X_list_q=list(X_q.columns)


        fi_q=[X_list_q,r_list_q]
        df_fi_q=pd.DataFrame (fi_q).transpose()
        df_fi_q.columns = ['Feature', 'Importance']
        df_fi_q=(df_fi_q.sort_values(by='Importance',ascending=True).reset_index(drop=True)).copy()

        fig_fi_q = px.bar(df_fi_q,
                            x="Importance", y="Feature",
                            template="plotly_white", title="Feature Importance",labels={"Feature":"Feature","Importance":"Importance"},width=300, height=400,
                            color_discrete_sequence=['#B2182B'], orientation='h')
        fig_fi_q.update_layout( xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),title_x=0.5)
        st.plotly_chart(fig_fi_q)

    with col26:
        #Filtering customers with churn rate higher than 0.5
        df_pareto = df_churn[df_churn['churn_probability'] > 0.5]

        #Sorting the dataaccording to monthly charges
        df_pareto=(df_pareto.sort_values(by='Monthly Charges' , ascending=False).reset_index(drop=True)).copy()
        #Calculating Prospective annual revenue loss
        df_pareto['Annual Revenue Loss']= df_pareto['Monthly Charges']*12

        #Cumulative values
        cumulative_sum=df_pareto['Annual Revenue Loss'].cumsum()
        total = df_pareto['Annual Revenue Loss'].sum()
        percentage = cumulative_sum / total * 100

        #Visual representaion of revenue loss with customer id
        trace1 = go.Bar(
            x=df_pareto['CustomerID'],
            y=df_pareto['Annual Revenue Loss'],
            marker=dict(
                color='rgb(178,24,43)'
                    ),
            name='Potential Annual Loss'
        )
        trace2 = go.Scatter(
            x=df_pareto['CustomerID'],
            y=percentage,
            marker=dict(
            color='rgb(255,255,0)'
                    ),
            name='Cumulative Percentage',
            yaxis='y2'

        )

        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pareto.add_trace(trace1)
        fig_pareto.add_trace(trace2,secondary_y=True)
        fig_pareto['layout'].update(height = 400, width = 500, title = "Pareto analysis of Annual Loss",xaxis=dict(tickangle=-90,showgrid=False),template="plotly_white", yaxis=dict(showgrid=False),title_x=0.5)
        st.plotly_chart(fig_pareto)    









    
    #Filtering customers with churn rate higher than 0.5
    df_test_3=df_test_2.copy()
    df_test_3 = df_test_2[df_test_2['churn_probability'] > 0.5]

    #Copying only customerid and monthly charges
    df_test_4=df_test_3[['CustomerID','Monthly Charges']].copy()
    #Sorting the dataaccording to monthly charges
    df_test_4=(df_test_4.sort_values(by='Monthly Charges' , ascending=False).reset_index(drop=True)).copy()
    #Calculating Prospective annual revenue loss
    df_test_4['Annual Revenue Loss']= df_test_4['Monthly Charges']*12

    #Cumulative values
    cumulative_sum=df_test_4['Annual Revenue Loss'].cumsum()
    total = df_test_4['Annual Revenue Loss'].sum()
    percentage = cumulative_sum / total * 100

    #Visual representaion of revenue loss with customer id
    trace1 = go.Bar(
        x=df_test_4['CustomerID'],
        y=df_test_4['Annual Revenue Loss'],
        marker=dict(
            color='rgb(34,163,192)'
                ),
        name='Potential Annual Loss'
    )
    trace2 = go.Scatter(
        x=df_test_4['CustomerID'],
        y=percentage,
        name='Cumulative Percentage',
        yaxis='y2'

    )

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(trace1)
    fig_pareto.add_trace(trace2,secondary_y=True)
    fig_pareto['layout'].update(height = 600, width = 800, title = "Pareto analysis of Annual Loss",xaxis=dict(tickangle=-90,showgrid=False),template="plotly_white", yaxis=dict(showgrid=False),title_x=0.5)
    fig_pareto
