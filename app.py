import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from fpdf import FPDF

# ======================================================
# 0️⃣ HELPER FUNCTION: PDF GENERATION
# ======================================================
def create_category_pdf(dataframe, category_name, threshold_info):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, f'Student Risk Report: {category_name}', 0, 1, 'C')
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f'Criteria: {threshold_info}', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(25, 10, "Student ID", 1)
    pdf.cell(20, 10, "Risk %", 1)
    pdf.cell(25, 10, "Attendance", 1)
    pdf.cell(20, 10, "GPA", 1)
    pdf.cell(100, 10, "Primary Risk Factor", 1)
    pdf.ln()

    # Table Rows
    pdf.set_font("Arial", size=10)
    for index, row in dataframe.iterrows():
        pdf.cell(25, 10, str(row['student_id']), 1)
        pdf.cell(20, 10, str(row['risk_score']), 1)
        pdf.cell(25, 10, str(row['attendance']), 1)
        pdf.cell(20, 10, str(row['prev_gpa']), 1)
        
        explanation = str(row['risk_explanation'])
        if len(explanation) > 55:
            explanation = explanation[:52] + "..."
        pdf.cell(100, 10, explanation, 1)
        pdf.ln()

    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ======================================================
# 1️⃣ APP CONFIGURATION & SIDEBAR
# ======================================================
st.set_page_config(page_title="Universal Student Risk System", layout="wide")

st.sidebar.title("⚙️ University Config")
st.sidebar.write("Customize the rules for your institution.")

st.sidebar.divider()
st.sidebar.subheader("🚨 Strict Thresholds")
min_attendance = st.sidebar.slider("Minimum Attendance Required (%)", 0, 100, 75)
min_gpa = st.sidebar.number_input("Minimum Passing GPA", 0.0, 10.0, 2.0, step=0.1)

st.sidebar.divider()
st.sidebar.subheader("⚖️ Risk Weighting")
st.sidebar.info("How important is each factor for the Risk Score?")

w_att = st.sidebar.slider("Weight: Attendance", 0, 100, 40)
w_gpa = st.sidebar.slider("Weight: GPA", 0, 100, 30)
w_marks = st.sidebar.slider("Weight: Marks (CA/Midterm)", 0, 100, 30)

# ======================================================
# 2️⃣ MAIN APP LOGIC
# ======================================================
st.title("🎓 Universal Student At-Risk Detection")
st.write(f"**Current Rules:** Attendance < {min_attendance}% | GPA < {min_gpa}")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if "filter_status" not in st.session_state:
        st.session_state["filter_status"] = "High Risk"

    # ======================================================
    # 3️⃣ DYNAMIC RISK CALCULATION (WEIGHTED)
    # ======================================================
    risk_features = ["attendance", "ca_marks", "midterm_marks", "prev_gpa"]
    risk_scaler = MinMaxScaler()
    
    # Fit scaler on original data
    risk_scaler.fit(df[risk_features])
    norm_data = risk_scaler.transform(df[risk_features])
    norm_df = pd.DataFrame(norm_data, columns=risk_features)
    
    # Risk Components (1 - normalized score)
    r_att = 1 - norm_df["attendance"]
    r_ca = 1 - norm_df["ca_marks"]
    r_mid = 1 - norm_df["midterm_marks"]
    r_gpa = 1 - norm_df["prev_gpa"]

    # Weighted Score Formula
    total_weight = w_att + w_gpa + w_marks + w_marks 
    
    weighted_score = (
        (r_att * w_att) + 
        (r_gpa * w_gpa) + 
        (r_ca * w_marks) + 
        (r_mid * w_marks)
    ) / total_weight

    df["risk_score"] = (weighted_score * 100).round(2)

    # ======================================================
    # 4️⃣ CLUSTERING
    # ======================================================
    features = df.drop(columns=["student_id", "risk_score"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Identify Risk Levels
    cluster_risk = df.groupby("cluster")["risk_score"].mean().sort_values(ascending=False).index
    high_cluster = cluster_risk[0]
    mod_cluster = cluster_risk[1]
    safe_cluster = cluster_risk[2]

    df["cluster_level"] = df["cluster"].map({
        high_cluster: "High Risk",
        mod_cluster: "Moderate Risk",
        safe_cluster: "Safe"
    })

    # ======================================================
    # 5️⃣ SMART EXPLANATION
    # ======================================================
    def generate_explanation(row):
        reasons = []
        if row['attendance'] < min_attendance:
            reasons.append(f"⚠️ Attendance < {min_attendance}%")
        if row['prev_gpa'] < min_gpa:
            reasons.append(f"⚠️ GPA < {min_gpa}")
        
        if not reasons:
            if row['ca_marks'] < df['ca_marks'].quantile(0.25):
                reasons.append("Low CA Marks")
            if row['midterm_marks'] < df['midterm_marks'].quantile(0.25):
                reasons.append("Low Midterm")
        
        return ", ".join(reasons) if reasons else "Safe"

    df["risk_explanation"] = df.apply(generate_explanation, axis=1)

    # ======================================================
    # 6️⃣ DASHBOARD UI
    # ======================================================
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", len(df))
    c2.metric("Violating Attendance", len(df[df['attendance'] < min_attendance]))
    c3.metric("Violating GPA", len(df[df['prev_gpa'] < min_gpa]))

    st.divider()

    # Circles
    col1, col2, col3 = st.columns(3)
    counts = df["cluster_level"].value_counts()
    
    def risk_circle(color, label):
        pct = round((counts.get(label, 0) / len(df)) * 100) if len(df) > 0 else 0
        return f"""<div style='text-align:center; color:{color}; font-size:24px; font-weight:bold;'>{pct}%</div>"""

    with col1:
        st.markdown(risk_circle("#ff4b4b", "High Risk"), unsafe_allow_html=True)
        if st.button("🔴 High Risk", use_container_width=True): st.session_state["filter_status"] = "High Risk"
    with col2:
        st.markdown(risk_circle("#ffa726", "Moderate Risk"), unsafe_allow_html=True)
        if st.button("🟡 Moderate", use_container_width=True): st.session_state["filter_status"] = "Moderate Risk"
    with col3:
        st.markdown(risk_circle("#2e7d32", "Safe"), unsafe_allow_html=True)
        if st.button("🟢 Safe", use_container_width=True): st.session_state["filter_status"] = "Safe"

    # Filtered Table
    st.subheader(f"📋 Student List: {st.session_state['filter_status']}")
    display_df = df[df["cluster_level"] == st.session_state["filter_status"]].sort_values(by="risk_score", ascending=False)

    def style_rows(row):
        color = '#ff4b4b' if "High" in row['cluster_level'] else '#ffa726' if "Moderate" in row['cluster_level'] else '#2e7d32'
        return [f'background-color: {color}80; color: white'] * len(row)

    st.dataframe(
        display_df[["student_id", "risk_score", "attendance", "prev_gpa", "risk_explanation", "cluster_level"]]
        .style.apply(style_rows, axis=1),
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # 7️⃣ WHAT-IF ANALYSIS (RESTORED & UPGRADED)
    # ======================================================
    st.divider()
    st.subheader("⚡ What-If Simulation")
    
    selected_student = None

    if not display_df.empty:
        # Select Student
        s_id = st.selectbox("Select Student for Simulation:", display_df["student_id"].values)
        selected_student = df[df["student_id"] == s_id].iloc[0]

        # Simulation UI
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            st.info(f"**Current Risk:** {selected_student['risk_score']}")
            st.write(f"**Cluster:** {selected_student['cluster_level']}")
            st.write(f"**Reason:** {selected_student['risk_explanation']}")
        
        with col_sim2:
            # Sliders to adjust values
            new_att = st.slider("Attendance %", 0, 100, int(selected_student["attendance"]))
            new_gpa = st.slider("GPA", 0.0, 10.0, float(selected_student["prev_gpa"]))
            new_ca = st.slider("CA Marks", 0, 100, int(selected_student["ca_marks"]))

            # 🚀 Re-Calculate Weighted Risk on the fly
            # 1. Create simulated dataframe row
            sim_row = pd.DataFrame([[new_att, new_ca, selected_student["midterm_marks"], new_gpa]], columns=risk_features)
            
            # 2. Normalize using the global scaler
            sim_norm = risk_scaler.transform(sim_row)
            
            # 3. Apply weights (Logic matches main app)
            s_r_att = 1 - sim_norm[0][0]
            s_r_ca = 1 - sim_norm[0][1]
            s_r_mid = 1 - (1 - (selected_student["midterm_marks"] - df["midterm_marks"].min()) / (df["midterm_marks"].max() - df["midterm_marks"].min())) # Approx normalization for midterm as it wasn't slider-ed
            # (Simplification: using original normalized value for midterm since we didn't make a slider for it to save space)
            s_r_mid = 1 - risk_scaler.transform(pd.DataFrame([selected_student[risk_features]], columns=risk_features))[0][2]
            s_r_gpa = 1 - sim_norm[0][3]

            sim_weighted_score = (
                (s_r_att * w_att) + 
                (s_r_gpa * w_gpa) + 
                (s_r_ca * w_marks) + 
                (s_r_mid * w_marks)
            ) / total_weight
            
            new_risk = (sim_weighted_score * 100).round(2)

            st.metric("Projected Risk Score", new_risk, delta=round(selected_student['risk_score'] - new_risk, 2))
            
            if new_risk < 50:
                st.success("✅ Projected Risk is Low!")
            elif new_risk < 75:
                st.warning("⚠️ Projected Risk is Moderate.")
            else:
                st.error("🔴 Projected Risk is High.")

    # ======================================================
    # 8️⃣ VISUALIZATION (RESTORED)
    # ======================================================
    st.divider()
    st.subheader("Analysis Visualization")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = df["cluster_level"].map({"High Risk": "red", "Moderate Risk": "orange", "Safe": "green"})
    
    # 1. Plot All Students
    ax.scatter(df["attendance"], df["prev_gpa"], c=colors, alpha=0.5, label="Students")
    
    # 2. Plot Selected Student (Big X)
    if selected_student is not None:
        ax.scatter(
            selected_student["attendance"], selected_student["prev_gpa"],
            c="black", s=200, marker="X", label=f"Selected ({selected_student['student_id']})"
        )
        # 3. (Optional) Plot Simulated Position
        # ax.scatter(new_att, new_gpa, c="blue", s=100, marker="*", label="Simulated")

    # 4. Draw Threshold Lines (Sidebar Settings)
    ax.axvline(x=min_attendance, color='blue', linestyle='--', label=f'Min Att ({min_attendance}%)')
    ax.axhline(y=min_gpa, color='purple', linestyle='--', label=f'Min GPA ({min_gpa})')
    
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("GPA")
    ax.set_title("Risk Landscape: Attendance vs GPA")
    ax.legend()
    
    st.pyplot(fig)

    # ======================================================
    # 9️⃣ DOWNLOAD REPORTS
    # ======================================================
    st.divider()
    st.subheader("📄 Download Reports")
    
    report_cat = st.selectbox("Select Category:", ["High Risk", "Moderate Risk", "Safe"])
    
    if st.button("Generate PDF"):
        r_df = df[df["cluster_level"] == report_cat].sort_values(by="risk_score", ascending=False)
        if not r_df.empty:
            criteria_text = f"Att < {min_attendance}% | GPA < {min_gpa}"
            pdf_data = create_category_pdf(r_df, report_cat, criteria_text)
            st.download_button("📥 Download PDF", pdf_data, f"{report_cat}_Report.pdf", "application/pdf")
        else:
            st.warning("No students in this category.")