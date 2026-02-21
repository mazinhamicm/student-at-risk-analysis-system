# 🎓 Student At-Risk Detection System

An interactive ML-powered dashboard to identify and analyze at-risk students using clustering and risk scoring.

## 🚀 Features

- KMeans Clustering for student segmentation
- Continuous Risk Score (0–100 scale)
- Explainable risk factors
- Interactive risk distribution dashboard
- What-if attendance simulation
- Cluster visualization
- PDF report generation per risk category

## 🧠 Tech Stack

- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- FPDF

## 📊 How It Works

1. Students are clustered using KMeans.
2. Risk levels are assigned based on average attendance.
3. A normalized inverted risk score is calculated.
4. Feature contributions determine primary risk factors.
5. Interactive dashboard allows filtering and simulation.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

## 👨‍💻 Author

*Mazin Hami C M*  
B.Tech Computer Science (AI & ML)  
Machine Learning Enthusiast  

GitHub: https://github.com/mazinhamicm

