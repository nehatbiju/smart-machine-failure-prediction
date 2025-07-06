import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

# 🔹 Load Dataset
file_path = "C:/Users/pvpra/OneDrive/Desktop/motor_failure_dataset.csv"
df = pd.read_csv(file_path)

# 🔹 Check for missing values and fill them
df.fillna(df.mean(), inplace=True)  

# 🔹 Feature Selection (Exclude the 'Failure' column)
X = df.drop(columns=["Failure"])  
y = df["Failure"]

# 🔹 Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Handle Class Imbalance Using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 🔹 Split Data into Train (70%), Validation (15%), and Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 🔹 Convert Data to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

# 🔹 Define Parameter Grid for Hyperparameter Tuning
xgb_param_grid = {
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "n_estimators": [100, 200, 300]
}

# 🔹 Perform Grid Search for XGBoost
xgb_base = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=2)
xgb_grid.fit(X_train, y_train)

# 🔹 Get Best XGBoost Model
best_xgb_params = xgb_grid.best_params_
print("✅ Best XGBoost Parameters:", best_xgb_params)

# 🔹 Train XGBoost with Early Stopping
best_xgb = xgb.train(
    best_xgb_params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, "validation")],
    early_stopping_rounds=20,
    verbose_eval=True
)

# 🔹 Make Predictions on Test Set
y_pred_xgb = best_xgb.predict(dtest)
y_pred_xgb = (y_pred_xgb > 0.5).astype(int)  # Convert probabilities to binary (0/1)

# 🔹 Evaluate XGBoost Performance
print("\n✅ Final XGBoost Performance:\n", classification_report(y_test, y_pred_xgb))
print("\n✅ Final XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# 🔹 Define Parameter Grid for Random Forest
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"]
}

# 🔹 Perform Grid Search for Random Forest
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=2)
rf_grid.fit(X_train, y_train)

# 🔹 Get Best Random Forest Model
best_rf = rf_grid.best_estimator_
print("\n✅ Best Random Forest Parameters:", rf_grid.best_params_)

# 🔹 Train Random Forest Model
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

# 🔹 Evaluate Random Forest Performance
print("\n✅ Final Random Forest Performance:\n", classification_report(y_test, y_pred_rf))
print("\n✅ Final Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# 🔹 Train Voting Classifier (Ensemble Model)
ensemble_model = VotingClassifier(estimators=[
    ("rf", best_rf)
], voting="soft")  # XGBoost is already trained using train(), so only RF in VotingClassifier

ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)

# 🔹 Evaluate Ensemble Model Performance
print("\n✅ Final Ensemble Model Performance:\n", classification_report(y_test, y_pred_ensemble))
print("\n✅ Final Ensemble Model Accuracy:", accuracy_score(y_test, y_pred_ensemble))
import joblib  # For saving and loading models

# 🔹 Save the XGBoost Model
best_xgb.save_model("xgb_model.json")  # Saves XGBoost as JSON

# 🔹 Save the Random Forest Model
joblib.dump(best_rf, "rf_model.pkl")  # Saves RF as a .pkl file

# 🔹 Save the Scaler (Needed for Future Predictions)
joblib.dump(scaler, "scaler.pkl")

print("✅ Models Saved Successfully!")
