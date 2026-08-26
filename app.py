import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL AND ENCODER
# ==============================

model = joblib.load("placement_model.pkl")
encoder = joblib.load("encoder.pkl")

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="centered"
)

# ==============================
# TITLE
# ==============================

st.title("🎓 Placement Prediction System")
st.write("Enter student details to predict placement probability.")

st.divider()

# ==============================
# INPUT SECTION
# ==============================

st.subheader("📋 Student Information")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenth_board = st.selectbox(
    "10th Board",
    ["State Board", "WBBSE", "CBSE", "ICSE"]
)

tenth_marks = st.number_input(
    "10th Marks",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

twelfth_board = st.selectbox(
    "12th Board",
    [
        "CBSE",
        "WBCHSE",
        "Other state Board",
        "Diploma",
        "ISE",
        "WBBSE",
        "Diploma board - MSBTE",
        "CISCE",
        "ISC",
        "MSBTE",
        "BSEB"
    ]
)

twelfth_marks = st.number_input(
    "12th Marks",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

stream_options = [
    "Mechanical Engineering",
    "Electronics and Communication Engineering",
    "Information Technology",
    "Computer Science in AIML",
    "Computer Science and Engineering",
    "Production Engineering",
    "Civil Engineering",
    "Electrical Engineering",
    "Computer Science in Data Science",
    "Electrical and Electronics Engineering",
    "IMsc Maths and Computing",
    "Computer Science and Design",
    "Electronics and Communication and Engineeing",
    "Chemical Engineering",
    "Electronics Engineering",
    "Electronic Engineering"
]

stream = st.selectbox(
    "Stream",
    stream_options
)

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

internship = st.selectbox(
    "Internship",
    ["Yes", "No"]
)

training = st.selectbox(
    "Training",
    ["Yes", "No"]
)

backlog = st.selectbox(
    "Backlog in 5th sem",
    ["Yes", "No"]
)

project = st.selectbox(
    "Innovative Project",
    ["Yes", "No"]
)

communication = st.slider(
    "Communication Level",
    min_value=1,
    max_value=5,
    value=3
)

technical_course = st.selectbox(
    "Technical Course",
    ["Yes", "No"]
)

# ==============================
# PREDICTION BUTTON
# ==============================

st.divider()

if st.button("🔮 Predict Placement", use_container_width=True):

    # Create student dataframe
    student_data = pd.DataFrame([{
        "Gender": gender,
        "10th board": tenth_board,
        "10th marks": tenth_marks,
        "12th board": twelfth_board,
        "12th marks": twelfth_marks,
        "Stream": stream,
        "Cgpa": cgpa,
        "Internships(Y/N)": internship,
        "Training(Y/N)": training,
        "Backlog in 5th sem": backlog,
        "Innovative Project(Y/N)": project,
        "Communication level": communication,
        "Technical Course(Y/N)": technical_course
    }])

    # ==============================
    # ENCODE CATEGORICAL DATA
    # ==============================

    categorical_columns = [
        "Gender",
        "10th board",
        "12th board",
        "Stream",
        "Internships(Y/N)",
        "Training(Y/N)",
        "Backlog in 5th sem",
        "Innovative Project(Y/N)",
        "Technical Course(Y/N)"
    ]

    student_encoded = encoder.transform(
        student_data[categorical_columns]
    )

    encoded_student_df = pd.DataFrame(
        student_encoded,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=student_data.index
    )

    # Remove original categorical columns
    student_numeric = student_data.drop(
        columns=categorical_columns
    )

    # Combine numerical + encoded data
    student_final = pd.concat(
        [student_numeric, encoded_student_df],
        axis=1
    )

    # ==============================
    # PREDICTION
    # ==============================

    prediction = model.predict(student_final)[0]

    probability = model.predict_proba(
        student_final
    )[0][1]

    probability_percent = probability * 100

    # ==============================
    # RESULT
    # ==============================

    st.subheader("📊 Prediction Result")

    if prediction == 1:

        st.success("🎉 Student is likely to be PLACED!")

        st.metric(
            "Placement Probability",
            f"{probability_percent:.2f}%"
        )

    else:

        st.error(" Student is likely to be NOT PLACED.")

        st.metric(
            "Placement Probability",
            f"{probability_percent:.2f}%"
        )

    # Probability bar
    st.progress(
        int(probability_percent)
    )

    st.caption(
        "Prediction is generated using the trained Random Forest model."
    )
