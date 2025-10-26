# ==========================================
# AIR QUALITY INDEX (AQI) PREDICTION SYSTEM
# ==========================================
# Implements methods from IEEE Paper (Akhatkulov & Yalgoshev, 2025)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD

import IPython.display as ipydisplay


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

# Generate dummy data for demonstration with more samples
np.random.seed(42)
n_samples = 500 # Increased number of samples
data = pd.DataFrame({
    'AQI': np.random.randint(0, 500, n_samples),
    'AirPollutionCategory': np.random.choice(['Good', 'Moderate', 'Unhealthy', 'Very Unhealthy', 'Hazardous'], n_samples),
    'AQI_Category': np.random.choice(['Good', 'Moderate', 'Unhealthy', 'Very Unhealthy', 'Hazardous'], n_samples), # Include AQI_Category
    'DateTime': pd.to_datetime(pd.date_range(start='1/1/2023', periods=n_samples, freq='D')),
    'Feature1': np.random.rand(n_samples) * 100,
    'Feature2': np.random.rand(n_samples) * 50,
    'Feature3': np.random.rand(n_samples) * 20,
    'Feature4': np.random.rand(n_samples) * 5
})

# Introduce random missing values in feature columns
for col in ['Feature1', 'Feature2', 'Feature3', 'Feature4']:
    data.loc[data.sample(frac=0.1).index, col] = np.nan


# Save dummy data to a CSV file
data.to_csv("air_quality_dataset.csv", index=False)

print("✅ Dataset Loaded Successfully!")
ipydisplay.display(data.head())

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================
# KNN Imputation for missing values
imputer = KNNImputer(n_neighbors=9)
# Explicitly select feature columns for imputation, excluding AQI
feature_cols_to_impute = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
imputed_values = imputer.fit_transform(data[feature_cols_to_impute])

# Assign the imputed values back to the original DataFrame using .loc
data.loc[:, feature_cols_to_impute] = imputed_values


# Define features and labels
# Exclude 'AQI' from the features
X = data.drop(columns=["AQI", "AQI_Category", "AirPollutionCategory", "DateTime"], errors='ignore')
y = data["AirPollutionCategory"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Label encoding (applied after splitting)
label_enc = LabelEncoder()
y_train_encoded = label_enc.fit_transform(y_train)
y_test_encoded = label_enc.transform(y_test)


# Normalization (Min-Max)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)


# ==========================================
# STEP 3: MODEL TRAINING (CLASSICAL MODELS)
# ==========================================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True)
}

metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train_encoded) # Use encoded training labels
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_encoded, y_pred) # Use encoded testing labels
    prec = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0) # Use encoded testing labels
    rec = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0) # Use encoded testing labels
    f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0) # Use encoded testing labels
    metrics[name] = [acc, prec, rec, f1]
    print(f"\n🔹 {name} RESULTS 🔹")
    print(classification_report(y_test_encoded, y_pred, target_names=label_enc.classes_)) # Use original class names

# ==========================================
# STEP 4: FEEDFORWARD NEURAL NETWORK (FNN)
# ==========================================
def build_fnn(input_dim, optimizer):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(len(label_enc.classes_), activation='softmax') # Use number of unique classes from label encoder
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

optimizers = {
    "Adam": Adam(learning_rate=0.001),
    "Adagrad": Adagrad(learning_rate=0.01),
    "RMSprop": RMSprop(learning_rate=0.001),
    "SGD(learning_rate=0.01)": SGD(learning_rate=0.01) # Changed key to string
}

for opt_name, opt in optimizers.items():
    fnn = build_fnn(X_train.shape[1], opt)
    history = fnn.fit(X_train, y_train_encoded, epochs=30, batch_size=32, verbose=0) # Use encoded training labels
    _, acc = fnn.evaluate(X_test, y_test_encoded, verbose=0) # Use encoded testing labels
    metrics[f"FNN_{opt_name}"] = [acc, None, None, None]
    print(f"✅ FNN ({opt_name}) Accuracy: {acc:.4f}")

# ==========================================
# STEP 5: VISUALIZATION
# ==========================================

# 1️⃣ Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(pd.DataFrame(X).corr(), cmap="coolwarm", annot=False)
plt.title("Pollutant Feature Correlation Heatmap")
plt.show()

# 2️⃣ Model Performance Comparison
metric_df = pd.DataFrame(metrics, index=["Accuracy", "Precision", "Recall", "F1-Score"]).T
metric_df["Accuracy"].plot(kind='bar', figsize=(10,6), title="Model Accuracy Comparison", color="teal")
plt.ylabel("Accuracy")
plt.show()

# 3️⃣ Confusion Matrix (Random Forest)
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train_encoded) # Use encoded training labels
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test_encoded, y_pred_best) # Let confusion_matrix infer labels

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 4️⃣ AQI Distribution
plt.figure(figsize=(8,5))
sns.histplot(data["AQI"], kde=True, bins=30, color='purple')
plt.title("Air Quality Index (AQI) Distribution")
plt.xlabel("AQI Value")
plt.ylabel("Frequency")
plt.show()

# 5️⃣ FNN Training Visualization
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title("FNN Training Progress")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ==========================================
# STEP 6: SUMMARY
# ==========================================
print("\n📊 Final Model Performance Summary:")
print(metric_df)