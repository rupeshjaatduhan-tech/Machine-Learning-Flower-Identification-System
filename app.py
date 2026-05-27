import streamlit as st
import numpy as np
import pickle
with open("iris_dataset.pkl",'rb') as f:
    model=pickle.load(f)

st.title("Iris Flower Prediction")

speal_length=st.slider("speal length in cm", 4.0,8.0)
speal_width=st.slider("speal width in cm", 4.0,8.0)
petal_length=st.slider("petal length in cm", 4.0,8.0)
petal_width=st.slider("petal width in cm", 4.0,8.0)

if st.button("prediction"):
    input_data=np.array([[speal_length,speal_width,petal_length,petal_width]])
    prediction = model.predict(input_data)
    species = ['Iris-setosa','Iris-Versicolor','Iris-Virginica']
    st.success(f"Predicted iris species:{species[int(prediction)]}")
