"""Student Performance Prediction Using Machine Learning (25CSA201).

Generates a synthetic 500-student dataset, trains Linear Regression and
Random Forest Regression, compares them, and predicts marks for a new student.

Run from the repository root:  python code/student_performance.py
Plots are saved to outputs/, the dataset to dataset/student_performance.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "student_performance.csv"
OUT_DIR = ROOT / "outputs"
DATA_PATH.parent.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ---- Generate synthetic dataset ------------------------------------------
np.random.seed(42)
n = 500
study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(40, 100, n)
previous_marks = np.random.uniform(30, 100, n)
noise = np.random.normal(0, 5, n)
final_marks = 2.5 * study_hours + 0.25 * attendance + 0.45 * previous_marks + noise
final_marks = np.clip(final_marks, 0, 100)

data = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Marks": previous_marks,
    "Final_Marks": final_marks,
})
data.to_csv(DATA_PATH, index=False)

print("First five records:")
print(data.head().round(2))
print("\nDataset shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())

# ---- Visualise ------------------------------------------------------------
for feature in ["Study_Hours", "Attendance", "Previous_Marks"]:
    plt.figure(figsize=(8, 5))
    plt.scatter(data[feature], data["Final_Marks"])
    plt.xlabel(feature)
    plt.ylabel("Final Marks")
    plt.title(feature + " vs Final Marks")
    plt.grid(True)
    plt.savefig(OUT_DIR / f"{feature.lower()}_vs_final_marks.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---- Split ----------------------------------------------------------------
X = data[["Study_Hours", "Attendance", "Previous_Marks"]]
y = data["Final_Marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ---- Train ----------------------------------------------------------------
linear_model = LinearRegression().fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

forest_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)


# ---- Evaluate -------------------------------------------------------------
def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    print("\nModel:", name)
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}


linear_result = evaluate_model("Linear Regression", y_test, linear_predictions)
forest_result = evaluate_model("Random Forest Regression", y_test, forest_predictions)

comparison = pd.DataFrame([linear_result, forest_result])
print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))
comparison.round(4).to_csv(OUT_DIR / "model_comparison.csv", index=False)

# ---- Select best model (lower MAE) ---------------------------------------
if linear_result["MAE"] <= forest_result["MAE"]:
    best_model, best_name = linear_model, "Linear Regression"
else:
    best_model, best_name = forest_model, "Random Forest Regression"
print("\nSelected model:", best_name)

predictions = best_model.predict(X_test)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.plot([0, 100], [0, 100], linestyle="--")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.grid(True)
plt.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Predict for a new student -------------------------------------------
new_student = pd.DataFrame({"Study_Hours": [6], "Attendance": [85], "Previous_Marks": [75]})
predicted_marks = float(np.clip(best_model.predict(new_student)[0], 0, 100))
print("\nNew student details:")
print(new_student)
print("Predicted marks:", round(predicted_marks, 2))

if predicted_marks >= 75:
    category = "Excellent"
elif predicted_marks >= 60:
    category = "Good"
elif predicted_marks >= 40:
    category = "Average"
else:
    category = "Needs Improvement"
print("Performance category:", category)
