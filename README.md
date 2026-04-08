# 🧠 Diabetes Prediction System (Deep Learning)

An end-to-end machine learning project that predicts the likelihood of diabetes using a neural network built with TensorFlow. The model is optimized, evaluated using multiple metrics, and enhanced with explainability using SHAP.

---

## 🚀 Features

* 🔹 Neural Network using TensorFlow/Keras
* 🔹 Feature Scaling with StandardScaler
* 🔹 Train/Test Split with Stratification
* 🔹 Early Stopping & Model Checkpointing
* 🔹 Evaluation using Accuracy, Precision, Recall, F1-score
* 🔹 ROC Curve & AUC Score
* 🔹 SHAP Explainability for feature importance

---

## 📊 Model Performance

* ✅ Accuracy: ~78%
* ✅ ROC-AUC Score: **0.83**

---

## 📈 ROC Curve

![ROC Curve](images/roc_curve.png)
---

## 🧪 Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Matplotlib
* SHAP

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python diabetes_model.py
```

---

## 📂 Dataset

* Pima Indians Diabetes Dataset
* Contains medical attributes such as glucose level, BMI, age, etc.

---

## 🧠 Key Learnings

* Built a complete ML pipeline from preprocessing to evaluation
* Improved model generalization using dropout and early stopping
* Used ROC-AUC for better performance evaluation
* Applied SHAP for model interpretability

---

## 📌 Future Improvements

* Deploy as a web app (Streamlit / FastAPI)
* Add hyperparameter tuning (KerasTuner)
* Improve accuracy using advanced architectures

---

## 👨‍💻 Author

Narein Karthik
