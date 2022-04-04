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

def user_input_features():
    df = pd.DataFrame({'smoker': ['Yes', 'No']}) 
    return df
df = user_input_features()


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

if smoker == 'yes':
    smoke = 1
else:
    smoke = 0

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

medcost = pd.read_csv("https://raw.githubusercontent.com/afiqahrupawon/myownweb/main/insurance.csv")
#smoker = {'no': 0,'yes': 1}
#medcost.smoker = [smoker[item] for item in medcost.smoker] 
X = medcost[['age','bmi','smoker']]
Y = medcost.charges

features = [smoke, age, bmi]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size =0.30,random_state=0)
#building the model
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

int_features = [int(x) for x in features]
final_features = [np.array(int_features)]


if st.button('Predict'):           # when the submit button is pressed
    prediction = model.predict(final_features)
    st.balloons()
    st.success(f'Your insurance charges would be: ${round(prediction[0],2)}')
