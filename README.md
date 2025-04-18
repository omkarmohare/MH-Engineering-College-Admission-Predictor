# 🎓 Engineering College Admission Predictor

A user-friendly web application designed to help engineering aspirants predict their chances of admission into engineering colleges in Maharashtra based on their MHT-CET scores. Built using machine learning algorithms like AdaBoost, this system assists students in making informed decisions by generating a customized list of suitable colleges.

---

## 📌 Project Overview

The Engineering College Admission Predictor is a data-driven system that takes in user input such as category, gender, percentile, and rank to recommend the most suitable engineering colleges. The system leverages past cutoff data and machine learning models to deliver accurate, personalized predictions.

---

## 🚀 Features

- 🧠 Machine Learning prediction using AdaBoost
- 📊 Data preprocessing and transformation
- 📈 Accuracy up to 91.74% using AdaBoost algorithm
- 🧾 PDF-to-CSV data conversion pipeline
- 🌐 Web interface built using Flask, HTML, CSS, Bootstrap
- 🔐 Clean, simple, and accessible user experience

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, Bootstrap  
- **Backend**: Python, Flask  
- **ML Libraries**: scikit-learn, Pandas, NumPy, Matplotlib  
- **Algorithms Used**:
  - AdaBoost (✅ Best accuracy: **91.74%**)
  - Linear Regression
  - Random Forest
  - Decision Tree

---

## 📁 File Structure


---

## 🧠 Machine Learning Model

- **Training Algorithm**: AdaBoost Classifier  
- **Train-Test Split**: 80:20  
- **Accuracy**: 91.74%  
- **F1 Score**: 0.88  
- **R² Score**: 0.93  

---

## 🎯 How it Works

1. Users input:
   - Category
   - Gender
   - Rank
   - Percentile
2. The input is sent to the backend Flask app.
3. The trained AdaBoost model predicts the best-fit engineering colleges.
4. The prediction results are displayed on the web interface.

---

## 📊 Results

| Algorithm         | Accuracy  |
|------------------|-----------|
| AdaBoost         | 91.74%    |
| Linear Regression| 70.12%    |
| Decision Tree    | 65%       |
| Random Forest    | 57.01%    |

---

## 👥 Team

- **Chaitanya Gabhane** (Exam Seat No. 202101070136)  
- **Abdul Muqueet Sahil** (Exam Seat No. 202101070183)  
- **Omkar Mohare** (Exam Seat No. 202101040236)  
- **Pragati Mundhe** (Exam Seat No. 202101050025)

### 👩‍🏫 Guide  
**Dr. Usha Verma**,  
MIT Academy of Engineering, Alandi (D), Pune

---

## 🔮 Future Scope

- Expand to include all Indian states & other entrance exams (e.g., JEE).
- Automate data collection from the DTE website.
- Improve model performance using advanced ML techniques like CatBoost or XGBoost.
- Integrate chatbot for student queries.

---

## 📚 Dataset Source

[MHT-CET 2024 Official Website](https://mhtcet2024.mahacet.org)

---

## 📝 License

This project is for academic purposes. Contact the authors for further use or collaboration.

