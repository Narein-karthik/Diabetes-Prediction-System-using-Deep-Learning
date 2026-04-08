import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
import tensorflow as tf
import numpy as np
...

np.random.seed(42)
tf.random.set_seed(42)
# ---------------- GPU SETUP ----------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU Detected: {gpus}")
else:
    print("ℹ️ Running on CPU")

# ---------------- LOAD DATA ----------------
try:
    dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
except OSError:
    print("❌ Dataset not found. Make sure CSV is in the same folder.")
    exit()

X = dataset[:, 0:8]
y = dataset[:, 8]

# ---------------- TRAIN/TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
def build_model(units=32, dropout=0.2):
    model = Sequential([
        Dense(units, activation='relu', input_shape=(8,)),
        Dropout(dropout),
        Dense(units // 2, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# create model
model = build_model(units=64, dropout=0.3)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- CALLBACKS ----------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

# ---------------- TRAIN ----------------
print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ---------------- EVALUATE ----------------
print("\n🎯 Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# ---------------- PREDICTIONS ----------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_probs = model.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("imgs/ROC_curve.png")
plt.show()

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------- SAMPLE OUTPUT ----------------
print("\n🔍 Sample Predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i][0]} | Actual: {int(y_test[i])}")

# ---------------- SAVE SCALER ----------------
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ Model + Scaler saved. Ready for deployment!")

import shap

print("\n🔍 Running SHAP explainability...")

# Use small sample (SHAP is slow)
sample_X = X_train[:100]

# Create explainer
explainer = shap.Explainer(model, sample_X)

# Explain test samples
shap_values = explainer(X_test[:50])

# Plot feature importance
shap.plots.bar(shap_values)
