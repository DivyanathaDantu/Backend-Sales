import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from MyUtils.HideStDefaults import hideNavBar
from MyUtils.searchAndSelectFile import selectDataset
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular

hideNavBar()


df = selectDataset()
chosen_target_x = st.multiselect(label="Choose Independant  variable", options=df.columns)
chosen_target_y = st.selectbox(label="Choose Dependant  variable",
                                   options=(df.columns).insert(0, "Choose an option"))

if chosen_target_y != 'Choose an option':
    #Training the model
    model=RandomForestClassifier()
    X = df[chosen_target_x]
    Y = df[chosen_target_y]
    model.fit(X,Y)

    #Extracting the column names of the independant variables
    X_featurenames=X.columns
    #X_featurenames

    #Defining the LIME class
    explainer = lime.lime_tabular.LimeTabularExplainer(
                        np.array(X),
                        feature_names=X_featurenames, 
                        class_names=['Does not Churn','Churn'], 
                        categorical_features=[0,1,2,3,4,5,6,7,8,9,10,11] ,                             
                        verbose=False, 
                                                )

    #search_cust = st.text_input("Search Your Customer ID")
    #identifier_value = st.selectbox(label="Choose Customer ID", options=df[(df["CustomerID"]>= search_cust) & (df["CustomerID"]< search_cust) ])
    identifier_value = st.selectbox(label="Choose Customer ID", options=df["CustomerID"])
    index = df[df["CustomerID"] == identifier_value].index
    i= index
    #st.write(i[0])
    exp1 = explainer.explain_instance(X.iloc[i[0]], 
    model.predict_proba, num_features=5)
    df_weights= pd.DataFrame(exp1.as_list())
    df_weights.rename( columns={0:'Factor',1:'Influence'}, inplace=True )
    df_weights

    fig= px.bar(df_weights,
                     x='Factor', y='Influence',
                     template="plotly_white", title="Impact on Customer Churn Probability",width=600, height=500)
    fig.update_layout( xaxis_title="Factors", yaxis_title="Percentage Contribution to Churn Probability")
    st.plotly_chart(fig)

    #Visual representaion of churn probability
    prob_array=model.predict_proba(X.iloc[2:3,:])

    churn_prob=prob_array[0][1]
    no_churn_prob=prob_array[0][0]

    column=['Churn','Will Not Churn']
    prob=[churn_prob,no_churn_prob]

    prob_dict={'Outcome':column,'Probability':prob}
    df_prob=pd.DataFrame(prob_dict)

    fig_prob= px.pie(df_prob,
                        names='Outcome', values='Probability',
                        template="plotly_white", title="Customer Churn Probability",width=600, height=500,color='Outcome',color_discrete_map={
                                    'Churn':'royalblue',
                                    'Will Nor Churn':'darkblue'})
    fig_prob.update_layout( xaxis_title="Factors", yaxis_title="Percentage Contribution to Churn Probability")
    st.plotly_chart(fig_prob)

    #df_queried=df.copy().reset_index(drop=True)
    
    #loading the input data into the variables
    #X=df_queried[chosen_target_x]
    #Y=df_queried[chosen_target_y]
    model_fi=RandomForestClassifier()
    model_fi.fit(X,Y)
    model_fi_=RandomForestClassifier()
    fi = permutation_importance(model, X, Y, n_repeats=10,random_state=0)

    r_list=list(fi.importances_mean)
    X_list=list(X.columns)
    fi=[X_list,r_list]
    df_fi=pd.DataFrame (fi).transpose()
    df_fi.columns = ['Feature', 'Importance']

    fig_fi = px.bar(df_fi,
                        x="Importance", y="Feature",
                        template="plotly_white", title="Feature Importance",labels={"Feature":"Feature","Importance":"Importance"},width=1000, height=500,
                        color_discrete_sequence=['#00CC96'], orientation='h')
    st.plotly_chart(fig_fi)

    query_variable=st.selectbox(label="Choose Query", options=df.columns)
    query_variable_value=st.selectbox(label="Choose Query", options=df[query_variable].unique())
    condition = (df[query_variable] == query_variable_value) 
    df_queried=(df[condition]).copy().reset_index(drop=True)
    U=df_queried[chosen_target_x]
    V=df_queried[chosen_target_y]
    model_fi=RandomForestClassifier()
    model_fi.fit(U,V)
    model_fi_=RandomForestClassifier()
    fi5 = permutation_importance(model, U, V, n_repeats=1,random_state=0)
    u_list=list(fi5.importances_mean)
    v_list=list(U.columns)
    fi5=[u_list,v_list]
    df_fi5=pd.DataFrame (fi5).transpose()
    df_fi5.columns = ['Feature', 'Importance']
    fig_fi5 = px.bar(df_fi5,
                     x="Importance", y="Feature",
                     template="plotly_white", title="Feature Importance",labels={"Feature":"Feature","Importance":"Importance"},width=1000, height=500,
                     color_discrete_sequence=['#00CC96'], orientation='h')
    st.plotly_chart(fig_fi5)
