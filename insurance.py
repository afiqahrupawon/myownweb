import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.write("""
#Medical Cost Prediction App
This app predicts the medical cost or insurance charges according to your attributes.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Your age', 18, 100)
    bmi = st.sidebar.slider('Your BMI', 15, 50)
    smoker = st.sidebar.slider('Are you a smoker? [Note: 0 for no, 1 for yes]', 0, 1)
    data = {'age': age,
            'bmi': bmi,
            'smoker': smoker}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

medcost = pd.read_csv("https://raw.githubusercontent.com/afiqahrupawon/myownweb/main/insurance.csv")
medcost['smoker'] = medcost['smoker'].map({'yes': 1, 'no': 0})
X = medcost[['age','bmi','smoker']]
Y = medcost.charges



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size =0.30,random_state=1)
#building the model
model = LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

prediction = model.predict(df)

st.subheader('Prediction')
st.write(medcost.charges_names[prediction])
#st.write(prediction)
