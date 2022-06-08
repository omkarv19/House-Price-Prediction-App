
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

model=pickle.load(open('Random_forest_regression_model.pkl','rb'))

df=pickle.load(open('df.pkl', 'rb'))

        
st.title("House Price Prediction App") 

Rooms=st.number_input('Rooms')
Type=st.selectbox('Type',df.Type.unique())
Method=st.selectbox('method',df.Method.unique())
Seller=st.selectbox('Seller',df.SellerG.unique())
Distance=st.number_input('Distance')
Car=st.number_input('Car')
Bathroom=st.number_input('Bathroom')
YearBuilt=st.slider('YearBuilt',min_value=min(df.YearBuilt),max_value=max(df.YearBuilt))
Landsize=st.number_input('Landsize')
BuildingArea=st.number_input('BuildingArea')
Postcode=st.slider('Postcode',min_value=min(df.Postcode),max_value=max(df.Postcode))

def Prediction(Rooms,Type,Method,Seller,Distance,Postcode,Bathroom,Car,Landsize,BuildingArea,YearBuilt):
                pred=model.predict([[Rooms,Type,Method,Seller,Distance,Postcode,Bathroom,Car,Landsize,BuildingArea,YearBuilt]])
                return pred

if st.button("Predict"): 
        result = Prediction(Rooms,Type,Method,Seller,Distance,Postcode,Bathroom,Car,Landsize,BuildingArea,YearBuilt) 
        exp_result=np.exp(result)
        st.success('Your Price is {}'.format(exp_result))
        





