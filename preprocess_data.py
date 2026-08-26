import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_placement_data.csv")

# Remove personal information
df = df.drop(columns=["Email", "Name"])

# Separate features and target
X = df.drop(columns=["Placement(Y/N)?"])

y = df["Placement(Y/N)?"].map({
    "Placed": 1,
    "Not Placed": 0
})

# Find categorical columns
categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("===== CATEGORICAL COLUMNS =====")
print(categorical_columns)

# One-Hot Encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_encoded = encoder.fit_transform(X[categorical_columns])
# Save the encoder for future predictions
joblib.dump(encoder, "encoder.pkl")

print("\n===== ENCODER SAVED =====")
print("Encoder saved as encoder.pkl")

# Convert encoded data into DataFrame
encoded_df = pd.DataFrame(
    X_encoded,
    columns=encoder.get_feature_names_out(categorical_columns),
    index=X.index
)

# Remove original categorical columns
X_numeric = X.drop(columns=categorical_columns)

# Combine numerical + encoded categorical features
X_final = pd.concat([X_numeric, encoded_df], axis=1)
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)
# 5-Fold Cross-Validation
cv_scores = cross_val_score(
    model,
    X_final,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n===== CROSS-VALIDATION =====")
print("Fold Scores:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train Random Forest
rf_model.fit(X_train, y_train)
# Save the trained Random Forest model
joblib.dump(rf_model, "placement_model.pkl")

print("\n===== MODEL SAVED =====")
print("Random Forest model saved as placement_model.pkl")
# Random Forest 5-Fold Cross-Validation
rf_cv_scores = cross_val_score(
    rf_model,
    X_final,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n===== RANDOM FOREST CROSS-VALIDATION =====")
print("Fold Scores:", rf_cv_scores)
print(f"Mean RF CV Accuracy: {rf_cv_scores.mean() * 100:.2f}%")

print("\n===== RANDOM FOREST TRAINED =====")
print("Random Forest model trained successfully!")
# Create Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Train Gradient Boosting
gb_model.fit(X_train, y_train)
# Gradient Boosting 5-Fold Cross-Validation
gb_cv_scores = cross_val_score(
    gb_model,
    X_final,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n===== GRADIENT BOOSTING CROSS-VALIDATION =====")
print("Fold Scores:", gb_cv_scores)
print(f"Mean GB CV Accuracy: {gb_cv_scores.mean() * 100:.2f}%")
# Compare model cross-validation scores
model_scores = {
    "Logistic Regression": cv_scores.mean(),
    "Random Forest": rf_cv_scores.mean(),
    "Gradient Boosting": gb_cv_scores.mean()
}

best_model_name = max(model_scores, key=model_scores.get)

print("\n===== MODEL COMPARISON =====")

for name, score in model_scores.items():
    print(f"{name}: {score * 100:.2f}%")

print("\n===== BEST MODEL =====")
print(f"Best Model: {best_model_name}")
print(f"CV Accuracy: {model_scores[best_model_name] * 100:.2f}%")

print("\n===== GRADIENT BOOSTING TRAINED =====")
print("Gradient Boosting model trained successfully!")

print("\n===== MODEL TRAINED =====")
print("Logistic Regression model trained successfully!")

print("\n===== TRAINING DATA =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n===== TESTING DATA =====")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL ACCURACY =====")
print(f"Accuracy: {accuracy * 100:.2f}%")
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)
# Random Forest predictions
rf_pred = rf_model.predict(X_test)

# Random Forest accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n===== RANDOM FOREST ACCURACY =====")
print(f"Accuracy: {rf_accuracy * 100:.2f}%")
# Random Forest evaluation
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\n===== RANDOM FOREST EVALUATION =====")
print(f"Precision: {rf_precision * 100:.2f}%")
print(f"Recall:    {rf_recall * 100:.2f}%")
print(f"F1 Score:  {rf_f1 * 100:.2f}%")
# Gradient Boosting predictions
gb_pred = gb_model.predict(X_test)

# Gradient Boosting accuracy
gb_accuracy = accuracy_score(y_test, gb_pred)

print("\n===== GRADIENT BOOSTING ACCURACY =====")
print(f"Accuracy: {gb_accuracy * 100:.2f}%")

print("\n===== ORIGINAL FEATURE SHAPE =====")
print(X.shape)

print("\n===== FINAL FEATURE SHAPE =====")
print(X_final.shape)

print("\n===== TARGET VALUES =====")
print(y.value_counts())
