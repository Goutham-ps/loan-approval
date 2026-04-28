import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Loan AI System", layout="wide")
st.title("🏦 AI Loan Approval System")



st.markdown("""
<style>

/* 🔥 Make whole header sticky */
.block-container {
    padding-top: 1rem;
}

/* Target main header area */
header, .stApp > div:first-child {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: #0e1117;
}

/* Sticky tabs */
div[data-testid="stTabs"] {
    position: sticky;
    top: 60px;  /* adjust based on title height */
    z-index: 998;
    background-color: #0e1117;
    border-bottom: 1px solid #333;
}

/* Optional: smooth look */
h1 {
    margin-bottom: 0px;
}

</style>
""", unsafe_allow_html=True)
# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

model = load_model()

# ----------------------------
# DATABASE SETUP
# ----------------------------
conn = sqlite3.connect("loan_app.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gender TEXT,
    married TEXT,
    dependents INTEGER,
    education TEXT,
    self_employed TEXT,
    applicant_income REAL,
    coapplicant_income REAL,
    loan_amount REAL,
    loan_term REAL,
    credit_history INTEGER,
    property_area TEXT,
    prediction INTEGER,
    probability REAL,
    timestamp TEXT
)
""")
conn.commit()

# ----------------------------
# TABS
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Insights", "📁 History"])

# ============================
# 🔮 TAB 1 — PREDICTION
# ============================
with tab1:

    st.sidebar.header("🔧 Applicant Details")

    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Married = st.sidebar.selectbox("Married", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
    Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

    ApplicantIncome = st.sidebar.slider("Applicant Income", 0, 50000, 5000)
    CoapplicantIncome = st.sidebar.slider("Coapplicant Income", 0, 30000, 0)
    LoanAmount = st.sidebar.slider("Loan Amount", 1, 500, 100)
    Loan_Amount_Term = st.sidebar.selectbox("Loan Term", [360, 180, 120, 60])
    Credit_History = st.sidebar.selectbox("Credit History", [1, 0])

    Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # ----------------------------
    # INPUT DATA
    # ----------------------------
    input_data = pd.DataFrame({
        'Gender': [Gender],
        'Married': [Married],
        'Dependents': [Dependents],
        'Education': [Education],
        'Self_Employed': [Self_Employed],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area]
    })

    # Feature engineering
    input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
    input_data['Income_per_Loan'] = input_data['TotalIncome'] / input_data['LoanAmount']

    st.subheader("📋 Applicant Summary")
    st.dataframe(input_data, use_container_width=True)

    # ----------------------------
    # PREDICTION
    # ----------------------------
    if st.button("🚀 Predict Loan Status"):

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("## 🎯 Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.success("Approved ✅")
            else:
                st.error("Rejected ❌")
        with col2:
            st.metric("Approval Probability", f"{prob*100:.2f}%")

        with col3:
            if prob > 0.8:
                st.success("🟢 Low Risk")
            elif prob > 0.5:
                st.warning("🟡 Medium Risk")
            else:
                st.error("🔴 High Risk")

        st.progress(float(prob))

        # ----------------------------
        # SAVE TO DATABASE
        # ----------------------------
        cursor.execute("""
        INSERT INTO predictions (
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_term, credit_history, property_area,
            prediction, probability, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Property_Area,
            int(prediction), float(prob),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()

# ============================
# 📊 TAB 2 — INSIGHTS
# ============================
with tab2:

    st.subheader("📊 Analytics Dashboard")

    # Load data
    data = pd.read_sql("SELECT * FROM predictions", conn)

    if len(data) == 0:
        st.info("No data available yet. Make some predictions first.")
    
    else:
        # ----------------------------
        # 🧠 ADD RISK COLUMN
        # ----------------------------
        def risk(p):
            return "Low" if p > 0.8 else "Medium" if p > 0.5 else "High"

        data['risk'] = data['probability'].apply(risk)

        # ----------------------------
        # 📊 METRICS
        # ----------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Applications", len(data))

        with col2:
            approval_rate = (data['prediction'].mean()) * 100
            st.metric("Approval Rate", f"{approval_rate:.2f}%")

        with col3:
            avg_income = data['applicant_income'].mean()
            st.metric("Avg Income", f"₹ {avg_income:.0f}")

        st.markdown("---")

        # ----------------------------
        # 📊 APPROVAL DISTRIBUTION
        # ----------------------------
        st.subheader("📊 Loan Approval Distribution")

        approval_counts = data['prediction'].value_counts().rename({
            1: "Approved",
            0: "Rejected"
        })

        st.bar_chart(approval_counts)

        # ----------------------------
        # 📈 INCOME DISTRIBUTION
        # ----------------------------
        st.subheader("📈 Income Distribution")

        st.bar_chart(data['applicant_income'])

        # ----------------------------
        # 📍 PROPERTY AREA ANALYSIS
        # ----------------------------
        st.subheader("📍 Property Area Analysis")

        area_counts = data['property_area'].value_counts()
        st.bar_chart(area_counts)

        # ----------------------------
        # 🧠 RISK DISTRIBUTION
        # ----------------------------
        st.subheader("🧠 Risk Distribution")

        risk_counts = data['risk'].value_counts()
        st.bar_chart(risk_counts)

        # ----------------------------
        # 🔥 TOP HIGH INCOME APPLICANTS
        # ----------------------------
        st.subheader("🔥 Top Applicants by Income")

        top_data = data.sort_values(by='applicant_income', ascending=False).head(10)
        st.dataframe(
            top_data[['applicant_income', 'loan_amount', 'prediction', 'risk']].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
# ============================
# 📁 TAB 3 — HISTORY + FILTERS
# ============================
with tab3:

    st.subheader("📁 Prediction History")

    data = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)

    if len(data) == 0:
        st.info("No predictions yet")

    else:
        st.markdown("### 🔍 Filters")

        col1, col2, col3 = st.columns(3)

        with col1:
            status_filter = st.selectbox("Loan Status", ["All", "Approved", "Rejected"])

        with col2:
            risk_filter = st.selectbox("Risk Level", ["All", "Low", "Medium", "High"])

        with col3:
            min_val = int(data['applicant_income'].min())
            max_val = int(data['applicant_income'].max())

            if min_val == max_val:
                st.info(f"Only one income value available: ₹ {min_val}")
                min_income, max_income = min_val, max_val
            else:
                min_income, max_income = st.slider(
                    "Income Range",
                    min_val,
                    max_val,
                    (min_val, max_val)
                )
        filtered = data.copy()

        if status_filter == "Approved":
            filtered = filtered[filtered['prediction'] == 1]
        elif status_filter == "Rejected":
            filtered = filtered[filtered['prediction'] == 0]

        def risk(p):
            return "Low" if p > 0.8 else "Medium" if p > 0.5 else "High"

        filtered['risk'] = filtered['probability'].apply(risk)

        if risk_filter != "All":
            filtered = filtered[filtered['risk'] == risk_filter]

        filtered = filtered[
            (filtered['applicant_income'] >= min_income) &
            (filtered['applicant_income'] <= max_income)
        ]

        st.dataframe(filtered, use_container_width=True)

        csv = filtered.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇️ Download Data",
            csv,
            "loan_history.csv",
            "text/csv"
        )
        st.caption(f"Total records: {len(data)}")