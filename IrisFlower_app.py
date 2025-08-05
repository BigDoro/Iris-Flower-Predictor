import streamlit as st
import pickle
import numpy as np
import pandas as pd
import scikit_learn as sklearn
from sklearn.datasets import load_iris

iris= load_iris()
df= pd.DataFrame(data=iris.data, columns=iris.feature_names)

st.title("Iris Flower Species Predictor")
st.write("Welcome! This app will predict the species of an Iris flower based on its features.")

model_list = {
    'Logistic Regression': 'Logistic_Regression_model.pkl',
    'Decision Tree Classifier': 'Decision_Tree_model.pkl',
    'Random Forest Classifier': 'Random_Forest_model.pkl',
    'K-Nearest Neighbors': 'KNN_model.pkl'
}

select_model = st.selectbox('Choose a model', list(model_list.keys()))
with open(model_list[select_model], 'rb') as f:
    model = pickle.load(f)

sepal_length = st.slider('Sepal Length (cm)', df['sepal length (cm)'].min(), df['sepal length (cm)'].max(), step=0.1)
sepal_width = st.slider('Sepal Width (cm)', df['sepal width (cm)'].min(), df['sepal width (cm)'].max(), step=0.1)
petal_length = st.slider('Petal Length (cm)', df['petal length (cm)'].min(), df['petal length (cm)'].max(), step=0.1)
petal_width = st.slider('Petal Width (cm)', df['petal width (cm)'].min(), df['petal width (cm)'].max(), step=0.1)

if st.button('Predict'):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)
    species = ['Setosa', 'Versicolor', 'Virginica']
    
    st.success(f'The predicted species is: {species[prediction[0]]}')
    
    probability_predict = model.predict_proba(data)[0]
    st.info(f'Model confidence: Setosa: {probability_predict[0] * 100:.2f}%, '
            f'Versicolor: {probability_predict[1] * 100:.2f}%, '
            f'Virginica: {probability_predict[2] * 100:.2f}%')



