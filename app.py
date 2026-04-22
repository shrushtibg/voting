# app.py

import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Title
st.title("🌳 Random Forest Voting Classifier")

# Load dataset (Iris for demo)
data = load_iris()
X = data.data
y = data.target

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Input Section
st.header("Enter Flower Features")

sepal_length = st.number_input("Sepal Length", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, step=0.1)

# Convert input to array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction button
if st.button("Predict"):
    
    # Get predictions from each tree (voting)
    tree_preds = []
    for tree in model.estimators_:
        pred = tree.predict(input_data)[0]
        tree_preds.append(pred)
    
    # Majority voting
    final_prediction = max(set(tree_preds), key=tree_preds.count)

    # Output
    st.subheader("🌲 Individual Tree Votes:")
    st.write(tree_preds)

    st.subheader("✅ Final Prediction (Majority Vote):")
    st.write(data.target_names[final_prediction])
