import joblib
import pandas as pd

# ==============================
# LOAD MODEL AND ENCODER
# ==============================

model = joblib.load("placement_model.pkl")
encoder = joblib.load("encoder.pkl")

print("===== MODEL LOADED =====")
print("Placement prediction system is ready!")
def get_marks(message):
    while True:
        try:
            value = float(input(message))

            if 0 <= value <= 100:
                return value

            print("❌ Marks 0 se 100 ke beech hone chahiye.")

        except ValueError:
            print("❌ Please valid number enter karo.")


def get_cgpa(message):
    while True:
        try:
            value = float(input(message))

            if 0 <= value <= 10:
                return value

            print("❌ CGPA 0 se 10 ke beech hona chahiye.")

        except ValueError:
            print("❌ Please valid CGPA enter karo.")


def get_communication(message):
    while True:
        try:
            value = int(input(message))

            if 1 <= value <= 5:
                return value

            print("❌ Communication Level 1 se 5 ke beech hona chahiye.")

        except ValueError:
            print("❌ Sirf 1, 2, 3, 4 ya 5 enter karo.")


def get_yes_no(message):
    while True:
        value = input(message).strip().lower()

        if value in ["yes", "no"]:
            return value.title()

        print("❌ Sirf Yes ya No enter karo.")


def get_stream(message):
    valid_streams = [
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

    while True:
        value = input(message).strip()

        for stream_name in valid_streams:
            if value.lower() == stream_name.lower():
                return stream_name

        print("❌ Invalid stream!")
        print("Please valid stream enter karo.")


# ==============================
# STUDENT INPUT
# ==============================

gender = input("Enter Gender (Male/Female): ").strip().title()

tenth_board = input("Enter 10th Board: ").strip()

tenth_marks = get_marks("Enter 10th Marks (0-100): ")

twelfth_board = input("Enter 12th Board: ").strip()

twelfth_marks = get_marks("Enter 12th Marks (0-100): ")

stream = get_stream("Enter Stream: ")

cgpa = get_cgpa("Enter CGPA (0-10): ")

internship = get_yes_no("Internship (Yes/No): ")

training = get_yes_no("Training (Yes/No): ")

backlog = get_yes_no("Backlog in 5th sem (Yes/No): ")

project = get_yes_no("Innovative Project (Yes/No): ")

communication = get_communication("Communication Level (1-5): ")

technical_course = get_yes_no("Technical Course (Yes/No): ")


# ==============================
# CREATE STUDENT DATA
# ==============================

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


print("\n===== STUDENT DATA =====")
print(student_data)


# ==============================
# CATEGORICAL COLUMNS
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


# ==============================
# ENCODE STUDENT DATA
# ==============================

student_encoded = encoder.transform(
    student_data[categorical_columns]
)

encoded_student_df = pd.DataFrame(
    student_encoded,
    columns=encoder.get_feature_names_out(categorical_columns),
    index=student_data.index
)


# ==============================
# COMBINE FEATURES
# ==============================

student_numeric = student_data.drop(
    columns=categorical_columns
)

student_final = pd.concat(
    [student_numeric, encoded_student_df],
    axis=1
)


# ==============================
# MAKE PREDICTION
# ==============================

prediction = model.predict(student_final)[0]

probability = model.predict_proba(student_final)[0][1]


# ==============================
# DISPLAY RESULT
# ==============================

print("\n================================")
print("      PLACEMENT PREDICTION")
print("================================")

if prediction == 1:
    print("Prediction: PLACED")
else:
    print("Prediction: NOT PLACED")

print(f"Placement Probability: {probability * 100:.2f}%")

print("================================")
