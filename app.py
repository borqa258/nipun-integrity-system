import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from datetime import datetime
import random

st.set_page_config(page_title="NIPUN Integrity Monitoring System", layout="wide")

# -----------------------------
# DATA GENERATION
# -----------------------------

@st.cache_data
def generate_data():

    np.random.seed(42)

    districts = [f"District {i}" for i in range(1, 11)]
    blocks = [f"Block {i}" for i in range(1, 51)]

    schools = []
    students = []
    assessments = []

    school_id = 0
    student_id = 0

    for d in districts:
        for b in random.sample(blocks, 10):
            for _ in range(10):
                school_id += 1
                enrollment = np.random.randint(80, 400)

                fln_prev = np.random.randint(20, 60)
                fln_current = fln_prev + np.random.randint(-5, 40)
                fln_current = max(5, min(fln_current, 100))

                attendance_avg = np.random.randint(70, 95)
                attendance_assessment = attendance_avg + np.random.randint(-15, 10)

                completion_speed = np.random.uniform(0.5, 3.0)

                schools.append([
                    school_id, d, b, enrollment,
                    fln_prev, fln_current,
                    attendance_avg,
                    attendance_assessment,
                    completion_speed
                ])

                for _ in range(enrollment):
                    student_id += 1
                    score = np.random.randint(0, 100)
                    students.append([student_id, school_id, score])

    school_df = pd.DataFrame(schools, columns=[
        "School_ID", "District", "Block", "Enrollment",
        "FLN_Previous", "FLN_Current",
        "Attendance_Avg_30D",
        "Attendance_Assessment_Day",
        "Completion_Speed"
    ])

    student_df = pd.DataFrame(students, columns=[
        "Student_ID", "School_ID", "FLN_Score"
    ])

    return school_df, student_df

school_df, student_df = generate_data()

# -----------------------------
# INTEGRITY MODEL
# -----------------------------

def compute_integrity(df):

    features = df[[
        "FLN_Previous",
        "FLN_Current",
        "Attendance_Avg_30D",
        "Attendance_Assessment_Day",
        "Completion_Speed"
    ]]

    model = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = model.fit_predict(features)
    df["Anomaly_Score"] = model.decision_function(features)

    df["Attendance_Variance"] = abs(df["Attendance_Avg_30D"] - df["Attendance_Assessment_Day"])
    df["FLN_Jump"] = df["FLN_Current"] - df["FLN_Previous"]

    df["Integrity_Index"] = (
        (df["FLN_Jump"] * 0.3) +
        (df["Attendance_Variance"] * 0.2) +
        (df["Completion_Speed"] * 10 * 0.2) +
        (np.where(df["Anomaly"] == -1, 30, 0))
    )

    df["Integrity_Risk_Score"] = np.clip(df["Integrity_Index"], 0, 100)
    df["Audit_Trigger"] = df["Integrity_Risk_Score"] > 60

    return df

school_df = compute_integrity(school_df)

# -----------------------------
# ROLE BASED ACCESS
# -----------------------------

st.sidebar.title("NIPUN Integrity Monitoring System")

role = st.sidebar.selectbox(
    "Role",
    ["State Admin", "District Officer", "Auditor", "Public View"]
)

district_filter = None

if role == "District Officer":
    district_filter = st.sidebar.selectbox("Select District", school_df["District"].unique())
    school_df = school_df[school_df["District"] == district_filter]

# -----------------------------
# NAVIGATION
# -----------------------------

page = st.sidebar.radio("Navigation", [
    "Overview",
    "School Explorer",
    "Integrity Engine",
    "Audit Management",
    "Examiner Analytics",
    "Forecasting",
    "Public Transparency"
])

# -----------------------------
# OVERVIEW
# -----------------------------

if page == "Overview":

    st.title("Statewide Integrity Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Schools", len(school_df))
    col2.metric("Avg FLN", round(school_df["FLN_Current"].mean(), 2))
    col3.metric("Flagged Schools", school_df["Audit_Trigger"].sum())
    col4.metric("Avg Integrity Risk", round(school_df["Integrity_Risk_Score"].mean(), 2))

    fig = px.bar(
        school_df.groupby("District")["FLN_Current"].mean().reset_index(),
        x="District",
        y="FLN_Current",
        title="District-wise FLN Performance"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(school_df, x="Integrity_Risk_Score", nbins=30)
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# SCHOOL EXPLORER
# -----------------------------

elif page == "School Explorer":

    st.title("School-Level Drilldown")

    school = st.selectbox("Select School", school_df["School_ID"])

    data = school_df[school_df["School_ID"] == school]
    students = student_df[student_df["School_ID"] == school]

    st.subheader("School Summary")
    st.dataframe(data)

    st.subheader("Student Score Distribution")
    fig = px.histogram(students, x="FLN_Score", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Integrity Risk Score", round(data["Integrity_Risk_Score"].values[0],2))
    st.metric("Audit Triggered", data["Audit_Trigger"].values[0])

# -----------------------------
# INTEGRITY ENGINE
# -----------------------------

elif page == "Integrity Engine":

    st.title("ML Integrity Engine")

    st.subheader("Top 20 High Risk Schools")
    st.dataframe(
        school_df.sort_values("Integrity_Risk_Score", ascending=False).head(20)
    )

    fig = px.scatter(
        school_df,
        x="FLN_Current",
        y="Integrity_Risk_Score",
        color="Audit_Trigger",
        size="Enrollment",
        hover_data=["District"]
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# AUDIT MANAGEMENT
# -----------------------------

elif page == "Audit Management":

    st.title("Audit Management")

    st.subheader("Mandatory Audit List")
    st.dataframe(school_df[school_df["Audit_Trigger"] == True])

    if st.button("Generate Random 5% Audit"):
        random_sample = school_df.sample(frac=0.05)
        st.dataframe(random_sample)

# -----------------------------
# EXAMINER ANALYTICS
# -----------------------------

elif page == "Examiner Analytics":

    st.title("Examiner Behavior Analytics")

    fig = px.box(school_df, y="Completion_Speed", title="Completion Speed Distribution")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(school_df, x="Attendance_Variance", nbins=20)
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# FORECASTING
# -----------------------------

elif page == "Forecasting":

    st.title("FLN Forecasting Model")

    X = school_df[["FLN_Previous"]]
    y = school_df["FLN_Current"]

    model = LinearRegression()
    model.fit(X, y)

    school_df["Predicted_FLN_Next_Cycle"] = model.predict(
        school_df[["FLN_Current"]]
    )

    fig = px.scatter(
        school_df,
        x="FLN_Current",
        y="Predicted_FLN_Next_Cycle",
        title="Predicted Next Cycle FLN"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PUBLIC TRANSPARENCY
# -----------------------------

elif page == "Public Transparency":

    st.title("Public Transparency Dashboard")

    public_view = school_df.groupby("District")[["FLN_Current", "Integrity_Risk_Score"]].mean().reset_index()
    st.dataframe(public_view)

    fig = px.bar(public_view, x="District", y="FLN_Current")
    st.plotly_chart(fig, use_container_width=True)

