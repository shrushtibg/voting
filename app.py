import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🗳️ Voting Participation Prediction")

st.write("Predict whether a person will participate in voting")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100)
education = st.selectbox("Education Level", ["School", "Graduate", "Postgraduate"])
employment = st.selectbox("Employment Status", ["Unemployed", "Employed"])
income = st.number_input("Monthly Income", min_value=0)
interest = st.slider("Interest in Politics (1-10)", 1, 10)
awareness = st.slider("Awareness Level (1-10)", 1, 10)

# Encoding
edu_map = {"School": 0, "Graduate": 1, "Postgraduate": 2}
emp_map = {"Unemployed": 0, "Employed": 1}

edu = edu_map[education]
emp = emp_map[employment]

# Dummy dataset (simple & stable)
X = np.array([
    [25,1,1,20000,7,8],
    [40,2,1,50000,9,9],
    [22,0,0,15000,3,4],
    [35,1,1,40000,6,7],
    [28,1,0,25000,5,5],
    [50,2,1,60000,10,9]
])

y = np.array([1,1,0,1,0,1])  # 1 = Vote, 0 = Not Vote

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, edu, emp, income, interest, awareness]])
    result = model.predict(input_data)

    if result[0] == 1:
        st.success("Likely to Vote 🗳️✅")
    else:
        st.error("Not Likely to Vote ❌")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("This app uses Random Forest to predict voting participation.")
