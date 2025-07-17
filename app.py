import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# UI inputs
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'Others', 'State-gov', 'Self-emp-inc', 'Federal-gov'])
fnlwgt = st.sidebar.number_input("Fnlwgt", 0, 1500000, 200000)
educational_num = st.sidebar.slider("Educational Num", 5, 16, 9)
marital_status = st.sidebar.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
occupation = st.sidebar.selectbox("Occupation", ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Others', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'])
relationship = st.sidebar.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
race = st.sidebar.selectbox("Race", ['Black', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 100000, 0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'England', 'China', 'Italy'])

# Build the input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Simplified encoding (in real case, use saved encoders)
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
sample_data = input_df.copy()
for col in categorical_cols:
    encoder = LabelEncoder()
    sample_vals = sample_data[col].unique().tolist()
    encoder.fit(sample_vals)
    input_df[col] = encoder.transform(input_df[col])

st.write("### 🔎 Encoded Input Data")
st.write(input_df)

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Encode categorical columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        encoder.fit(batch_data[col].astype(str))
        batch_data[col] = encoder.transform(batch_data[col].astype(str))

    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
