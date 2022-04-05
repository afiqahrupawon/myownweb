import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle



# load the model from disk
loaded_model = pickle.load(open('model_pkl', 'rb'))



# Titles 
st.title("Medical cost (Insurance) Charges Prediction App")
st.header("This app will calculate the (insurance) charges based on a person's attributes")

readme = st.checkbox("readme first")

if readme:

    st.write("""
        This web app is a demo using [streamlit](https://streamlit.io/) library. It is hosted on [heroku](https://www.heroku.com/). You may get the codes via [github](https://github.com/afiqahrupawon/myownweb)
        """)
    st.write ("The prediction for this web app is done using Linear Regression model. Please note that this model might not result in the best prediction")
    
    st.write ("For more info, please contact:")

    st.write("<a href='https://www.linkedin.com/in/nurul-afiqah-462777233/'> Nurul Afiqah Rupawon </a>", unsafe_allow_html=True)



def load_data():
    df = pd.DataFrame({'sex': ['Male','Female'],
                       'smoker': ['Yes', 'No']}) 
    return df
df = load_data()



def load_data():
    df1 = pd.DataFrame({'region' : ['southeast' ,'northwest' ,'southwest' ,'northeast']}) 
    return df1
df1 = load_data()



# Take the users input

sex = st.selectbox("Please select your gender", df['sex'].unique())
smoker = st.selectbox("Are you a smoker?", df['smoker'].unique())
region = st.selectbox("Please select your region", df1['region'].unique())
age = st.slider("Your age", 18, 100)
bmi = st.slider("Your BMI", 10, 50)
children = st.slider("Number of children", 0, 15)

# converting text input to numeric to get back predictions from backend model.
if sex == 'male':
    gender = 0
else:
    gender = 1
    
if smoker == 'yes':
    smoke = 1
else:
    smoke = 0
    
if region == 'northeast':
    reg = 3
elif region == 'northwest':
    reg = 2
elif region == 'southeast':
    reg = 1
else:
    reg = 0
    

# store the inputs
features = [gender, smoke, reg, age, bmi, children]
# convert user inputs into an array fr the model

int_features = [int(x) for x in features]
final_features = [np.array(int_features)]



if st.button('Predict'):      # when the submit button is pressed
    prediction =  loaded_model.predict(final_features)
    st.balloons()
    st.success(f'Your insurance charges is predicted to be: ${round(prediction[0],2)}')
