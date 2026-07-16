import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/student_model.pkl")

print("===== Student Performance Prediction =====")

age = int(input("Age: "))
gender = int(input("Gender (0=Male, 1=Female): "))
ethnicity = int(input("Ethnicity (0-3): "))
parental_education = int(input("Parental Education (0-4): "))
study_time = float(input("Study Time Weekly (hours): "))
absences = int(input("Absences: "))
tutoring = int(input("Tutoring (0=No,1=Yes): "))
parental_support = int(input("Parental Support (0-4): "))
extracurricular = int(input("Extracurricular (0=No,1=Yes): "))
sports = int(input("Sports (0=No,1=Yes): "))
music = int(input("Music (0=No,1=Yes): "))
volunteering = int(input("Volunteering (0=No,1=Yes): "))
gpa = float(input("Current GPA: "))

student = pd.DataFrame([[
    age,
    gender,
    ethnicity,
    parental_education,
    study_time,
    absences,
    tutoring,
    parental_support,
    extracurricular,
    sports,
    music,
    volunteering,
    gpa
]], columns=[
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "GPA"
])


# Grade Class Mapping
# 0 = Excellent
# 1 = Good
# 2 = Average
# 3 = Below Average
# 4 = Poor

prediction = model.predict(student)[0]

grade_map = {
    0.0: "Excellent",
    1.0: "Good",
    2.0: "Average",
    3.0: "Below Average",
    4.0: "Poor"
}

print("\n========== RESULT ==========")
print("Predicted Grade Class:", prediction)
print("Performance:", grade_map[prediction])