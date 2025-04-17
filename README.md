# MH-Engineering College Admission Predictor 🎓📊

This is a web-based application that predicts suitable engineering colleges in Maharashtra based on MHT-CET exam inputs. It uses a machine learning model (AdaBoost) trained on past admission data to suggest a list of colleges likely to grant admission.

---

## 🚀 Features

- Predicts eligible engineering colleges based on CET performance
- Takes multiple inputs: percentile, category, gender, and rank
- Easy-to-use web interface
- Built with Flask (Python backend) and AdaBoost ML model
- Returns a list of colleges you are likely to get admission into

---

## 🧠 Machine Learning Model

- **Algorithm**: AdaBoost Classifier
- **Training Data**: Historical admission data of Maharashtra engineering colleges
- **Inputs (Features)**:
  - CET Percentile
  - CET Rank
  - Category (e.g., Open, SC, ST, OBC)
  - Gender
- **Output**: Predicted list of eligible engineering colleges

---

## 🛠 Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **ML Tools**: Scikit-learn, Pandas, NumPy, AdaBoost
- **Deployment**: *(Optional - fill in if deployed)*

---

## 🔧 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/omkarmohare/MH-Engineering-College-Admission-Predictor.git
   cd MH-Engineering-College-Admission-Predictor
