import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🗳️ Voting Prediction App (Random Forest)")

st.write("Predict whether a person will vote")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Monthly Income", min_value=0)
education = st.selectbox("Education Level", ["School", "Graduate", "Postgraduate"])
employment = st.selectbox("Employment Status", ["Unemployed", "Employed", "Self-Employed"])
political_interest = st.slider("Political Interest (1-10)", 1, 10)
past_voting = st.selectbox("Voted in Previous Election?", ["Yes", "No"])

# Convert categorical to numeric
edu_map = {"School": 0, "Graduate": 1, "Postgraduate": 2}
emp_map = {"Unemployed": 0, "Employed": 1, "Self-Employed": 2}

edu = edu_map[education]
emp = emp_map[employment]
past_vote = 1 if past_voting == "Yes" else 0

# Dummy dataset
X = np.array([
    [25,20000,1,1,7,1],
    [40,50000,2,1,8,1],
    [22,15000,0,0,3,0],
    [35,40000,1,2,6,1],
    [28,25000,1,1,5,0],
    [50,60000,2,1,9,1]
])

y = np.array([1,1,0,1,0,1])  # 1 = Vote, 0 = Not Vote

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Prediction
if st.button("Predict Voting Behavior"):
    input_data = np.array([[age, income, edu, emp, political_interest, past_vote]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Will Vote 🗳️✅")
    else:
        st.error("Will Not Vote ❌")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("Random Forest classification model for voting prediction.")
