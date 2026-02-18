import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from fpdf import FPDF
import base64

# ======================================================
# 0️⃣ HELPER FUNCTION: PDF GENERATION
# ======================================================
def create_category_pdf(dataframe, category_name):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, f'Student Risk Report: {category_name}', 0, 1, 'C')
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
    pdf.cell(30, 10, "Student ID", 1)
    pdf.cell(25, 10, "Risk Score", 1)
    pdf.cell(25, 10, "Attendance", 1)
    pdf.cell(25, 10, "CA Marks", 1)
    pdf.cell(85, 10, "Primary Risk Factor", 1)
    pdf.ln()

    # Table Rows
    pdf.set_font("Arial", size=10)
    for index, row in dataframe.iterrows():
        pdf.cell(30, 10, str(row['student_id']), 1)
        pdf.cell(25, 10, str(row['risk_score']), 1)
        pdf.cell(25, 10, str(row['attendance']), 1)
        pdf.cell(25, 10, str(row['ca_marks']), 1)
        
        # Truncate explanation if too long for the cell
        explanation = str(row['risk_explanation'])
        if len(explanation) > 40:
            explanation = explanation[:37] + "..."
        pdf.cell(85, 10, explanation, 1)
        pdf.ln()

    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ======================================================
# 1️⃣ APP CONFIGURATION
# ======================================================
st.set_page_config(page_title="Student At-Risk Detection", layout="centered")

if "filter_status" not in st.session_state:
    st.session_state["filter_status"] = "High Risk"

st.title("Student At-Risk Detection System")
st.write("Upload student academic data to identify at-risk students.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head(), use_container_width=True)

    # ======================================================
    # 2️⃣ CLUSTERING & PRE-PROCESSING
    # ======================================================
    features = df.drop(columns=["student_id"])

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Identify Risk Levels
    cluster_risk_rank = df.groupby("cluster")["attendance"].mean().sort_values().index
    
    high_risk_cluster = cluster_risk_rank[0]
    moderate_risk_cluster = cluster_risk_rank[1]
    safe_cluster = cluster_risk_rank[2]

    def assign_label(c):
        if c == high_risk_cluster:
            return "High Risk"
        elif c == moderate_risk_cluster:
            return "Moderate Risk"
        else:
            return "Safe"

    df["cluster_level"] = df["cluster"].apply(assign_label)

    # ======================================================
    # 3️⃣ RISK SCORE CALCULATION
    # ======================================================
    risk_features = ["attendance", "ca_marks", "midterm_marks", "prev_gpa"]
    risk_scaler = MinMaxScaler()

    normalized = risk_scaler.fit_transform(df[risk_features])
    norm_df = pd.DataFrame(normalized, columns=risk_features)

    risk_component = 1 - norm_df
    df["risk_score"] = (risk_component.mean(axis=1) * 100).round(2)

    # Store components
    df["risk_attendance"] = risk_component["attendance"]
    df["risk_ca"] = risk_component["ca_marks"]
    df["risk_midterm"] = risk_component["midterm_marks"]
    df["risk_gpa"] = risk_component["prev_gpa"]

    # ======================================================
    # 4️⃣ RISK EXPLANATION
    # ======================================================
    def generate_risk_reason(row):
        contributions = {
            "Low attendance": row["risk_attendance"],
            "Low CA marks": row["risk_ca"],
            "Poor midterm performance": row["risk_midterm"],
            "Low previous GPA": row["risk_gpa"]
        }
        sorted_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top = [k for k, v in sorted_factors[:2] if v > 0.25]
        return ", ".join(top) if top else "Multiple moderate factors"

    df["risk_explanation"] = df.apply(generate_risk_reason, axis=1)

    # ======================================================
    # 5️⃣ INTERACTIVE RISK CIRCLES
    # ======================================================
    st.divider()
    st.subheader("Risk Distribution Dashboard")
    st.write("Click a button below the circle to filter the list.")

    total_students = len(df)
    counts = df["cluster_level"].value_counts()
    
    high_count = counts.get("High Risk", 0)
    mod_count = counts.get("Moderate Risk", 0)
    safe_count = counts.get("Safe", 0)

    if total_students > 0:
        high_pct = round((high_count / total_students) * 100)
        mod_pct = round((mod_count / total_students) * 100)
        safe_pct = round((safe_count / total_students) * 100)
    else:
        high_pct = mod_pct = safe_pct = 0

    col1, col2, col3 = st.columns(3)

    def risk_circle_html(color, percentage, label):
        return f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <div style="
                width: 120px; height: 120px; 
                border-radius: 50%; 
                background-color: {color}; 
                color: white; 
                font-size: 28px; 
                font-weight: bold; 
                display: flex; align-items: center; justify-content: center;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);">
                {percentage}%
            </div>
            <p style="margin-top: 10px; font-weight: bold; font-size: 16px; color: {color};">{label}</p>
        </div>
        """

    with col1:
        st.markdown(risk_circle_html("#ff4b4b", high_pct, "High Risk"), unsafe_allow_html=True)
        if st.button("Show High Risk 🔴", use_container_width=True):
            st.session_state["filter_status"] = "High Risk"

    with col2:
        st.markdown(risk_circle_html("#ffa726", mod_pct, "Moderate Risk"), unsafe_allow_html=True)
        if st.button("Show Moderate 🟡", use_container_width=True):
            st.session_state["filter_status"] = "Moderate Risk"

    with col3:
        st.markdown(risk_circle_html("#2e7d32", safe_pct, "Safe"), unsafe_allow_html=True)
        if st.button("Show Safe 🟢", use_container_width=True):
            st.session_state["filter_status"] = "Safe"

    # ======================================================
    # 6️⃣ FILTERED TABLE DISPLAY
    # ======================================================
    current_filter = st.session_state["filter_status"]
    st.markdown(f"### 📋 Student List: {current_filter}")

    filtered_df = df[df["cluster_level"] == current_filter].copy()

    display_df = (
        filtered_df[["student_id", "attendance", "ca_marks",
                     "midterm_marks", "prev_gpa",
                     "risk_score", "risk_explanation", "cluster_level"]]
        .sort_values(by="risk_score", ascending=False)
    )

    def style_risk_rows(row):
        level = row["cluster_level"]
        if "High Risk" in level:
            color = 'background-color: #ff4b4b; color: white;' 
        elif "Moderate" in level:
            color = 'background-color: #ffa726; color: black;'
        else:
            color = 'background-color: #2e7d32; color: white;'
        return [color] * len(row)

    if not display_df.empty:
        styled_df = display_df.style.apply(style_risk_rows, axis=1)
        st.dataframe(
            styled_df,
            column_config={
                "cluster_level": None,
                "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0, max_value=100),
                "student_id": st.column_config.TextColumn("ID"),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No students found in this category.")

    # ======================================================
    # 7️⃣ WHAT-IF ANALYSIS
    # ======================================================
    st.divider()
    st.subheader("What-If Analysis")

    selected_student = None 

    if not display_df.empty:
        selected_student_id = st.selectbox(
            "Select a student from the list above:",
            display_df["student_id"].values
        )

        selected_student = df[df["student_id"] == selected_student_id].iloc[0]

        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Current Risk Score:** {selected_student['risk_score']}")
        with col_b:
            st.write(f"**Main Factors:** {selected_student['risk_explanation']}")

        attendance_boost = st.slider("Simulate Attendance Increase (%)", 0, 20, 0)

        simulated = selected_student.copy()
        simulated["attendance"] = min(100, simulated["attendance"] + attendance_boost)
        sim_norm = risk_scaler.transform(pd.DataFrame([simulated[risk_features]]))
        sim_risk_component = 1 - pd.DataFrame(sim_norm, columns=risk_features)
        new_risk_score = round(sim_risk_component.mean(axis=1).iloc[0] * 100, 2)

        st.metric("Projected Risk Score", new_risk_score, delta=round(selected_student['risk_score'] - new_risk_score, 2))
    else:
        st.warning("Select a different category above to perform analysis.")

    # ======================================================
    # 8️⃣ VISUALIZATION
    # ======================================================
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = df["cluster_level"].map({
        "High Risk": "#ff4b4b", "Moderate Risk": "#ffa726", "Safe": "#2e7d32"
    })

    scatter = ax.scatter(
        df["attendance"], df["ca_marks"], 
        c=colors, alpha=0.4, s=50, label="Students"
    )

    if selected_student is not None:
        ax.scatter(
            selected_student["attendance"], selected_student["ca_marks"],
            color="black", s=200, marker='X', linewidths=2, edgecolors='white',
            label=f"Selected ({selected_student['student_id']})"
        )
        ax.annotate(
            f"ID: {selected_student['student_id']}",
            (selected_student["attendance"], selected_student["ca_marks"]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=9, fontweight='bold'
        )

    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("CA Marks")
    ax.set_title("Student Risk Clusters")
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4b4b', label='High Risk'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffa726', label='Moderate Risk'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2e7d32', label='Safe'),
    ]
    if selected_student is not None:
        legend_elements.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='black', label='Selected Student', markersize=10))

    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ======================================================
    # 9️⃣ DOWNLOAD REPORTS (NEW FEATURE)
    # ======================================================
    st.divider()
    st.subheader("📄 Download Reports")
    st.write("Generate and download a PDF list of students for a specific risk category.")

    # Dropdown to choose category
    report_category = st.selectbox(
        "Select Category to Download:",
        ["High Risk", "Moderate Risk", "Safe"]
    )

    if st.button("Generate PDF Report"):
        # Filter data for the report
        report_df = df[df["cluster_level"] == report_category].sort_values(by="risk_score", ascending=False)
        
        if not report_df.empty:
            # Generate PDF
            pdf_data = create_category_pdf(report_df, report_category)
            
            # Create download button
            st.success(f"✅ Report for **{report_category}** generated! ({len(report_df)} students)")
            st.download_button(
                label="📥 Click to Download PDF",
                data=pdf_data,
                file_name=f"Student_Report_{report_category.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
        else:
            st.warning(f"No students found in the **{report_category}** category.")