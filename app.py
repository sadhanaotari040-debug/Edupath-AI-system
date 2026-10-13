import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="EduPath AI",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 EduPath AI")
st.subheader("Student Educational Growth & Decision Support")
st.write("Enter your study details to get a Growth Score, Growth Level, Weak Area and personalized daily recommendation.")

@st.cache_data
def load_data():
    return pd.read_csv("student_performance_dataset.csv")

@st.cache_resource
def train_model():
    df = load_data().copy()
    df["Pass_Fail"] = df["Pass_Fail"].map({"Pass": 1, "Fail": 0})

    encoded = pd.get_dummies(
        df,
        columns=[
            "Parental_Education_Level",
            "Internet_Access_at_Home",
            "Extracurricular_Activities"
        ],
        drop_first=True
    )

    X = encoded.drop(
        columns=["Student_ID", "Gender", "Final_Exam_Score", "Pass_Fail"]
    )
    y = encoded["Pass_Fail"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns.tolist()

model, model_columns = train_model()

st.markdown("---")
st.header("📝 Enter Your Details")

study_hours = st.number_input(
    "Study Hours per Week",
    min_value=0.0, max_value=80.0, value=20.0, step=1.0
)

attendance = st.number_input(
    "Attendance Rate (%)",
    min_value=0.0, max_value=100.0, value=75.0, step=1.0
)

past_score = st.number_input(
    "Past Exam Score (%)",
    min_value=0.0, max_value=100.0, value=60.0, step=1.0
)

parent_education = st.selectbox(
    "Parental Education Level",
    ["High School", "Bachelor's", "Master's", "PhD"]
)

internet = st.selectbox(
    "Internet Access at Home",
    ["Yes", "No"]
)

extra = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

if st.button("🚀 Analyze My Growth", use_container_width=True):
    growth_score = (
        (study_hours / 40) * 30 +
        (attendance / 100) * 30 +
        (past_score / 100) * 40
    )
    growth_score = round(growth_score, 2)

    if growth_score >= 80:
        growth_level = "Excellent"
    elif growth_score >= 70:
        growth_level = "Good"
    elif growth_score >= 60:
        growth_level = "Average"
    else:
        growth_level = "Needs Improvement"

    weak_areas = []
    if study_hours < 20:
        weak_areas.append("Study Hours")
    if attendance < 75:
        weak_areas.append("Attendance")
    if past_score < 60:
        weak_areas.append("Past Exam Performance")

    weak_area = ", ".join(weak_areas) if weak_areas else "No Major Weak Area"

    recommendations = []
    if study_hours < 20:
        recommendations.append("Increase study time")
    if attendance < 75:
        recommendations.append("Improve attendance")
    if past_score < 60:
        recommendations.append("Practice previous exam topics")

    if growth_score >= 80:
        recommendations.append("Start advanced learning and projects")
    elif growth_score >= 70:
        recommendations.append("Continue regular practice and build projects")
    else:
        recommendations.append("Focus on fundamentals and consistency")

    recommendation = " | ".join(recommendations)

    # Prepare one input row in the same format used for model training.
    input_row = pd.DataFrame([{
        "Study_Hours_per_Week": study_hours,
        "Attendance_Rate": attendance,
        "Past_Exam_Scores": past_score,
        "Parental_Education_Level": parent_education,
        "Internet_Access_at_Home": internet,
        "Extracurricular_Activities": extra
    }])

    input_encoded = pd.get_dummies(
        input_row,
        columns=[
            "Parental_Education_Level",
            "Internet_Access_at_Home",
            "Extracurricular_Activities"
        ],
        drop_first=True
    )
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = int(model.predict(input_encoded)[0])
    predicted_result = "Pass" if prediction == 1 else "Fail"

    st.markdown("---")
    st.header("📊 Your Result")

    c1, c2 = st.columns(2)
    c1.metric("Growth Score", f"{growth_score}/100")
    c2.metric("Predicted Result", predicted_result)

    st.success(f"Growth Level: **{growth_level}**")
    st.info(f"Weak Area: **{weak_area}**")
    st.warning(f"Daily Recommendation: **{recommendation}**")

    st.caption("This tool is an educational project and should be used as guidance, not as an official academic decision.")
