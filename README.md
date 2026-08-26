# Placement Prediction System

An AI/ML-based Placement Prediction System that predicts whether a student is likely to be placed based on academic performance, skills, internships, training, projects, communication level, and other placement-related factors.

## Project Overview

This project uses machine learning classification algorithms to predict student placement outcomes.

The system includes:

* Data preprocessing
* One-Hot Encoding
* Logistic Regression
* Random Forest
* Gradient Boosting
* 5-Fold Cross-Validation
* Model evaluation
* Placement probability prediction
* Streamlit-based web interface

## Machine Learning Models

Three classification models were evaluated:

| Model               | Cross-Validation Accuracy |
| ------------------- | ------------------------: |
| Logistic Regression |                    88.53% |
| Random Forest       |                    89.28% |
| Gradient Boosting   |                    89.03% |

### Best Model

**Random Forest**

Cross-Validation Accuracy: **89.28%**

## Model Performance

The final test performance achieved:

* Accuracy: **90.12%**
* Precision: **88.10%**
* Recall: **92.50%**
* F1 Score: **90.24%**

### Confusion Matrix

```text
[[36  5]
 [ 3 37]]
```

## Features Used

The model uses the following student-related features:

* Gender
* 10th Board
* 10th Marks
* 12th Board
* 12th Marks
* Stream
* CGPA
* Internship
* Training
* Backlog in 5th Semester
* Innovative Project
* Communication Level
* Technical Course

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit

## Project Structure

```text
Placement-Prediction-System/
│
├── dataset/
│   └── cleaned_placement_data.csv
│
├── preprocess_data.py
├── predict.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-link>
cd Placement-Prediction-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python preprocess_data.py
```

This generates the trained model and encoder files.

### 4. Run prediction

```bash
python predict.py
```

### 5. Run Streamlit Application

```bash
streamlit run app.py
```

## Dataset

The project uses a cleaned student placement dataset containing academic, skill, training, internship, and project-related information.

## Disclaimer

This system provides a machine learning-based prediction and should be used for educational and analytical purposes. The prediction is not a guarantee of actual placement.
