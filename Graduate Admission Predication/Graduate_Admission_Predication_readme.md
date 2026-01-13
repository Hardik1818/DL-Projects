# Graduate Admission Prediction using Machine Learning

## 📌 Project Overview
Graduate admission is a competitive process where students are evaluated based on multiple academic and profile-related factors. This project aims to **predict the probability of a student getting admitted** into a graduate program using **Machine Learning techniques**.

The model helps students understand how different factors such as **GRE score, TOEFL score, CGPA, research experience, and university rating** affect admission chances.

---

## 🎯 Problem Statement
Students often struggle to evaluate their chances of getting admitted to a university. Manual evaluation is subjective and varies across institutions.

This project:
- Predicts **admission probability**
- Uses historical admission data
- Provides a **data-driven and objective estimation**

---

## 🧠 Solution Approach
We treat this problem as a **regression task**, where:
- Input → Student academic profile
- Output → Probability of admission (between 0 and 1)

A machine learning model is trained to learn the relationship between student features and admission outcomes.

---

## 🗂️ Dataset Description
The dataset contains information about applicants, including:

- **GRE Score**
- **TOEFL Score**
- **University Rating**
- **Statement of Purpose (SOP)**
- **Letter of Recommendation (LOR)**
- **CGPA**
- **Research Experience**
- **Chance of Admit** (Target variable)

The target variable is a **continuous value**, representing admission probability.

---

## ⚙️ Technologies & Libraries Used
- **Python**
- **NumPy** – numerical operations  
- **Pandas** – data manipulation  
- **Matplotlib & Seaborn** – data visualization  
- **Scikit-learn** – preprocessing & modeling  

---

## 🔄 Data Preprocessing
### Steps performed:
1. **Data loading and inspection**
2. **Checking for missing values**
3. **Feature selection**
4. **Feature scaling**
   - Standardization ensures all features are on the same scale
5. **Train-Test Split**
   - Training set for learning
   - Testing set for evaluation

---

## 📊 Exploratory Data Analysis (EDA)
EDA is performed to:
- Understand feature distributions
- Analyze correlations between features
- Visualize relationships between CGPA, GRE, and admission chances

This helps in selecting relevant features and improving model performance.

---

## 🏗️ Model Building
Depending on the notebook, common models used include:
- **Linear Regression**
- **Multiple Linear Regression**
- **Other regression-based ML models**

The model learns how each feature contributes to the final admission probability.

---

## 🧪 Model Evaluation
### Metrics Used:
- **R² Score**
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

These metrics help evaluate how close the predictions are to actual values.

---

## 🔮 Predictions
The trained model:
- Takes a new student profile as input
- Predicts the **probability of admission**
- Helps students make informed decisions about applications

---

## 📈 Results & Observations
- CGPA and GRE score have strong influence on admission chances
- Research experience improves probability
- Feature scaling significantly improves regression performance

---

## ✅ Advantages
- Simple and interpretable model
- Useful for students and counselors
- Fast training and prediction

---

## ⚠️ Limitations
- Based on historical data
- Does not account for qualitative factors like interviews
- Performance depends on dataset quality

---

## 🌍 Real-World Applications
- University admission counseling
- Student self-assessment tools
- Education analytics platforms

---

## 🛠️ Future Improvements
- Use advanced models (Random Forest, XGBoost)
- Add more real-world features
- Build a web application for live prediction
- Deploy using Flask or Streamlit

---

## 📌 Conclusion
This project demonstrates how **machine learning regression models** can be used to predict graduate admission chances effectively. It is suitable for:
- Academic projects
- Machine learning practice
- Portfolio and GitHub showcase

---

## 👨‍💻 Author
**Hardik Thapaliya**

---

⭐ If you find this project helpful, consider giving it a star!
