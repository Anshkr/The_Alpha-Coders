import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import lightgbm as lgb
import joblib
import numpy as np

# =======================
# 1. LOAD DATA
# =======================
train = pd.read_csv("ecommerce_fraud_train.csv")
test = pd.read_csv("ecommerce_fraud_test.csv")

# IDs for submission
test_ids = test.index.copy()

# Columns to drop
DROP_COLS = ["fraud_prob_hidden", "user_id"]

for col in DROP_COLS:
    if col in train.columns:
        train = train.drop(columns=[col])
    if col in test.columns:
        test = test.drop(columns=[col])

# Target split
y = train["is_fraud"]
X = train.drop(columns=["is_fraud"])

# =======================
# 2. COLUMN TYPES
# =======================
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# =======================
# 3. IMPUTATION
# =======================
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")

X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
test[numerical_cols] = num_imputer.transform(test[numerical_cols])

X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
test[categorical_cols] = cat_imputer.transform(test[categorical_cols])

# =======================
# 4. TARGET ENCODING
# =======================
encoder = TargetEncoder(cols=categorical_cols, smoothing=0.2)
X_encoded = encoder.fit_transform(X, y)
test_encoded = encoder.transform(test)

# =======================
# 5. SCALING
# =======================
ALL_FEATURES = X_encoded.columns.tolist()
scaler = StandardScaler()

X_encoded[ALL_FEATURES] = scaler.fit_transform(X_encoded[ALL_FEATURES])
test_encoded[ALL_FEATURES] = scaler.transform(test_encoded[ALL_FEATURES])

# =======================
# 6. TRAIN-VAL SPLIT
# =======================
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# 7. TRAIN LIGHTGBM
# =======================
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    class_weight={0: 1, 1: 30},
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='f1',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

joblib.dump(model, "fraud_model.pkl")
joblib.dump(num_imputer, "num_imputer.pkl")
joblib.dump(cat_imputer, "cat_imputer.pkl")
joblib.dump(encoder, "target_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(ALL_FEATURES, "model_columns.pkl")

# =======================
# 8. VALIDATION SCORE
# =======================
val_pred = model.predict(X_val)
print("🔥 Validation F1 Score:", f1_score(y_val, val_pred))

# =======================
# 9. FINAL PREDICTION
# =======================
test_pred = model.predict(test_encoded)

submission = pd.DataFrame({
    "id": test_ids,
    "fraud": test_pred.astype(int)
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")
