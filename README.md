# 🎓 Educational Resource Optimizer (EdTech Triage Engine)

[![Streamlit App]](https://student-at-risk-analysis-system.streamlit.app/)

A predictive Operations Research and Machine Learning engine designed for higher education administrations. This B2B SaaS prototype moves beyond basic "pass/fail" alerts by mathematically triaging at-risk students and automatically allocating constrained university resources (Counselors and Tutors) to maximize student retention and accreditation ROI (e.g., NAAC/NBA standards).

---

## 🚨 The Institutional Problem
Universities lose millions of dollars annually to student attrition, and current Academic Alert Systems are fundamentally broken due to two massive flaws:
1. **Static Thresholds & "The Boiling Frog" Problem:** Traditional systems only flag students *after* they cross a failure threshold (e.g., "Alert when GPA drops below 2.0"). They completely miss high-performing students who are suddenly exhibiting pre-dropout behavior.
2. **The Resource Bottleneck:** If an alert system flags 500 students, but the university only has 3 Academic Counselors, the administration is paralyzed. There is no mathematical way to prioritize *who* gets the counselor first to maximize the survival rate.

---

## 💡 The Novel Solution
This system replaces outdated threshold alerts with **Predictive Triage** and **Algorithmic Resource Allocation**. It doesn't just predict failure; it calculates the exact trajectory of the student and assigns limited staff directly to the most critical cases. 

### 🔥 Key Innovations & Features
* **Temporal Trajectory Predictor (The "Critical Drop" Alarm):** Instead of just looking at current attendance, the engine calculates a 2-week rolling velocity. If a student's performance drops by a critical margin (e.g.10%), the system flags a "Critical Drop," catching rapid behavioral crashes weeks before the student actually fails.
* **Knapsack-Inspired Resource Optimizer:** Deans input their exact staff constraints (e.g., Max 2 Counselors, 1 Tutor). The algorithm ranks the student body by absolute risk and automatically assigns human resources to the highest-ROI cases, dynamically pushing the rest to an automated Waitlist.
* **Interactive What-If Simulator:** A live projection tool allowing administrators to adjust a student's metrics (Attendance, CA Marks) and instantly run the data through the ML scaler to see the projected drop in their final Risk Score.
* **Algorithmic PDF Reporting:** Generates color-coded, heavily filtered institutional reports using `fpdf`. Deans can instantly export targeted lists of only the "Critical Drop" students for immediate emergency outreach.

---

## 🧠 Machine Learning & Algorithmic Architecture
This engine utilizes a multi-step mathematical pipeline to process raw student data into actionable triage directives:

1. **Temporal Velocity Penalty (Custom Math Model):** - Calculates the differential between present and past behavior: `Velocity = Current_Att - Past_Att`.
   - Applies an asymmetric mathematical penalty to the base risk score ($Penalty = |Velocity| \times 0.8$) strictly for negative trajectories, ensuring sudden behavioral drops aggressively spike the student's risk profile.
2. **Feature Normalization (`MinMaxScaler`):** - Transforms highly varied data points (Attendance percentages, GPA on a 10-point scale, and raw exam marks) into a uniform mathematical space to accurately calculate multi-variable base risk.
3. **Unsupervised Stratification (`K-Means Clustering`):** - Deploys a K-Means algorithm (`n_clusters=3`) to autonomously group the multidimensional student data into natural tiers: **High Risk**, **Moderate Risk**, and **Safe**, eliminating human bias in risk categorization.
4. **Rule-Based Triage Logic:** - Evaluates the student's specific cluster against their feature weaknesses (e.g., Low GPA vs. Critical Drop) to determine the *Ideal Intervention* (Dean Meeting, Counselor, Tutor, or Mentor).

---

## 💻 Tech Stack
* **Frontend UI & Cloud Deployment:** Streamlit
* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning Engine:** Scikit-Learn (MinMaxScaler, KMeans)
* **Visualization:** Matplotlib
* **Document Generation:** FPDF

---

## ⚙️ How to Run Locally
1.
pip install -r requirements.txt
streamlit run app.py
GitHub: https://github.com/mazinhamicm/student-at-risk-analysis-system
2.Streamlit: https://student-at-risk-analysis-system.streamlit.app/
