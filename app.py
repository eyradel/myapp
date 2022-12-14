import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier


import time

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            </style>
            <html><body><p></p><body/></html>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("https://delaeyram.com")
def linear():
    st.header("DATA MODELLING APPLICATION")
    file = st.file_uploader("Load Your Dataset",['csv','xlsx'])
    if file is not None:


        try:
            data = pd.read_excel(file)
        except:
            data = pd.read_csv(file)
        col1,col2 =  st.columns(2)
        with col1:
            if st.checkbox("View(Excel files)"):
                st.text(data)
        with col2:
            if st.checkbox("View Dataframe(csv files)"):
                st.dataframe(data)
        cell1,cell2 = st.columns(2)
        with cell1:
            x1 = st.multiselect("X",data.columns)
            x = data[x1]


        with cell2:
            y1 = st.multiselect("Y",data.columns)
            y = data[y1]



    if st.checkbox("Predict"):
        reg = LinearRegression()
        reg.fit(x,y)
        y_pred =reg.predict(x)
        st.header("Coefficient")
        st.text(reg.coef_)
        st.header("Intercept")
        st.text(reg.intercept_)
        st.text(r2_score(x,y_pred))











        if st.checkbox("plot"):
            fig, ax = plt.subplots()
            plt.title("LinearRegression")
            ax.scatter(x,y)
            ax.scatter(x,y_pred)

            st.pyplot(fig)


def multi():
    file = st.file_uploader("Load Your Dataset",['csv','xlsx'])
    if file is not None:


        try:
            data = pd.read_excel(file)
        except:
            data = pd.read_csv(file)
        col1,col2 =  st.columns(2)
        with col1:
            if st.checkbox("View(Excel files)"):
                st.text(data)
        with col2:
            if st.checkbox("View Dataframe(csv files)"):
                st.dataframe(data)
        cell1,cell2 = st.columns(2)
        with cell1:
            x1 = st.multiselect("X",data.columns)

            x = data[x1]



        with cell2:
            y1 = st.multiselect("Y",data.columns)
            y = data[y1]



    if st.checkbox("Predict"):
        reg = LinearRegression()
        reg.fit(x,y)
        y_pred =reg.predict(x)
        st.header("Coefficient")
        st.text(reg.coef_)
        st.header("Intercept")
        st.text(reg.intercept_)
    if st.checkbox("Accuracy"):

        st.metric("Score",r2_score(y,y_pred),1)


    if st.checkbox("plot"):
        fig, ax = plt.subplots()
        plt.title("plot")
        ax.scatter(x,y)
        ax.scatter(x,y_pred)

        st.pyplot(fig)






def tree():
     st.header("Tree")
     file = st.file_uploader("Load Your Dataset",['csv','xlsx'])
     if file is not None:


         try:
             data = pd.read_excel(file)
         except:
             data = pd.read_csv(file)
         col1,col2 =  st.columns(2)
         with col1:
             if st.checkbox("View(Excel files)"):
                 st.text(data)
         with col2:
             if st.checkbox("View Dataframe(csv files)"):
                 st.dataframe(data)
         cell1,cell2 = st.columns(2)
         with cell1:

             x1 = st.multiselect(" X select only two columns",data.columns)

             x = data[x1]



         with cell2:
             y1 = st.multiselect("Y",data.columns)
             y = data[y1]

     final  =  st.checkbox("Predict")
     if final:
        model = DecisionTreeClassifier()
        model.fit(x,y)


        st.markdown("input values to be predicted")
        g1,g2  = st.columns(2)
        with g1:
            ft = st.text_input("Input the first parameter values")
        with g2:
            sc = st.text_input("Input the second parameter values")
     fore = st.checkbox("forecast")
     if fore:
         predict = model.predict([[ft,sc]])
         st.markdown(predict)





st.sidebar.image("logo.png")
option = st.sidebar.selectbox("select",['Tree','Regression'])
if option=='Tree':
    tree()
elif option=='Regression':
    multi()
