import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
pipe = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('dataframe.pkl' , 'rb'))

# Front End
st.title("Diabetes Predictor")
Pregnancies = st.selectbox( 'Pregnancies', df['Pregnancies'].unique())
Glucose = st.selectbox('Glucose', df['Glucose'].unique())
BloodPressure = st.selectbox('BloodPressure', df['BloodPressure'].unique())
SkinThickness = st.selectbox('SkinThickness', df['SkinThickness'].unique())
Insulin = st.selectbox('Insulin', df['Insulin'].unique())
BMI = st.selectbox('BMI', df['BMI'].unique())
DiabetesPedigreeFunction = st.selectbox('DiabetesPedigreeFunction', df['DiabetesPedigreeFunction'].unique())
Age = st.selectbox('Age', df['Age'].unique())



if st.button('Predict Diabetes'):

    # Prepare the input data in the same format the model expects
    user_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    # Reshape it into a 2D array as the model expects (samples x features)
    user_data = user_data.reshape(1, -1)

    # Create a DataFrame with the same column names as the training data
    user_data_df = pd.DataFrame(user_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Get the predicted price from the model
    Predicted_Diabetes = pipe.predict(user_data_df)

    # Show the predicted price
    st.subheader(f"The predicted Diabetes is: {Predicted_Diabetes[0]:,.2f}")