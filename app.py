import streamlit as st
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

st.title("Voting Prediction App")

# Inputs
age = st.number_input("Age", min_value=18)
income = st.number_input("Income", min_value=0)
education = st.selectbox("Education", ["School", "Graduate", "Postgraduate"])
employment = st.selectbox("Employment", ["Unemployed", "Employed", "Self-Employed"])
interest = st.slider("Political Interest", 1, 10)
past_vote = st.selectbox("Voted Before?", ["Yes", "No"])

# Convert
edu_map = {"School": 0, "Graduate": 1, "Postgraduate": 2}
emp_map = {"Unemployed": 0, "Employed": 1, "Self-Employed": 2}

edu = edu_map[education]
emp = emp_map[employment]
past = 1 if past_vote == "Yes" else 0

# Dummy data
X = np.array([
    [25,20000,1,1,7,1],
    [40,50000,2,1,8,1],
    [22,15000,0,0,3,0],
    [35,40000,1,2,6,1]
])

y = np.array([1,1,0,1])

model = RandomForestClassifier()
model.fit(X, y)

# Prediction
if st.button("Predict"):
    data = np.array([[age, income, edu, emp, interest, past]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("Will Vote")
    else:
        st.error("Will Not Vote")
