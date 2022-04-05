import streamlit as st
import pandas as pd
import numpy as np
import pickle


# load the model from disk
loaded_model = pickle.load(open('model_pkl', 'rb'))



# Titles 
st.title("Medical Cost (Insurance) Charges Prediction App")
st.header("This app will calculate the (insurance) charges based on a person's attributes")

readme = st.checkbox("readme first")

if readme:

    st.write("""
        This web app is a demo using [streamlit](https://streamlit.io/) library. It is hosted on [heroku](https://www.heroku.com/). You may get the codes via [github](https://github.com/afiqahrupawon/myownweb)
        """)
    st.write ("The prediction for this web app is done using Linear Regression model. Please note that this model might not result in the best prediction")
    
    st.write ("For more info, please contact:")

    st.write("<a href='https://www.linkedin.com/in/nurul-afiqah-462777233/'> Nurul Afiqah Rupawon </a>", unsafe_allow_html=True)




# Take the users input

age = st.slider("Your age", 18, 100)
sex = st.slider("Please select your gender [0 is for 'Male', 1 is for 'Female']", 0, 1) 
bmi = st.slider("Your BMI", 10, 50)
children = st.slider("Number of children", 0, 15)
smoker = st.selectbox("Please select if you are a smoker or not [0 is for 'No', 1 is for 'Yes']", 0, 1)
region = st.selectbox("Please select your region [0 is for 'southwest', 1 is for 'southeast', 2 is for'northwest', 1 is for 'northeast' ]", 0, 3)

    

# store the inputs
features = [age, sex, bmi, children, , smoker, region]

# convert user inputs into an array fr the model
int_features = [int(x) for x in features]
final_features = [np.array(int_features)]



if st.button('Predict'):      # when the submit button is pressed
    prediction =  loaded_model.predict(final_features)
    st.balloons()
    st.success(f'Your insurance charges is predicted to be: ${round(prediction[0],2)}')
