import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from fpdf import FPDF
import google.generativeai as genai

# ======================================================
# 0️⃣ HELPER FUNCTION: PDF GENERATION (WITH COLORS)
# ======================================================
def create_category_pdf(dataframe, category_name, threshold_info):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 8, f'Institutional Resource Allocation: {category_name}', 0, 1, 'C')
            self.set_font('Arial', 'I', 9)
            self.cell(0, 8, f'Criteria: {threshold_info}', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(0, 0, 0) 
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    
    # --- Table Header ---
    pdf.set_fill_color(220, 220, 220) 
    pdf.set_text_color(0, 0, 0)       
    pdf.set_font("Arial", 'B', 8)
    
    pdf.cell(12, 10, "ID", 1, 0, 'C', fill=True)
    pdf.cell(15, 10, "Risk %", 1, 0, 'C', fill=True)
    pdf.cell(12, 10, "Att %", 1, 0, 'C', fill=True)
    pdf.cell(10, 10, "GPA", 1, 0, 'C', fill=True)
    pdf.cell(61, 10, "Reason & Trend", 1, 0, 'C', fill=True)
    pdf.cell(80, 10, "Algorithmic Allocation", 1, 1, 'C', fill=True) 
    
    # --- Table Rows ---
    pdf.set_font("Arial", size=8)
    for index, row in dataframe.iterrows():
        cluster = str(row.get('cluster_level', ''))
        
        # Apply the exact RGB colors from your dashboard UI
        if "High" in cluster:
            pdf.set_fill_color(255, 75, 75)   
            pdf.set_text_color(255, 255, 255) 
        elif "Moderate" in cluster:
            pdf.set_fill_color(255, 167, 38)  
            pdf.set_text_color(255, 255, 255) 
        elif "Safe" in cluster:
            pdf.set_fill_color(46, 125, 50)   
            pdf.set_text_color(255, 255, 255) 
        else:
            pdf.set_fill_color(255, 255, 255) 
            pdf.set_text_color(0, 0, 0)       

        pdf.cell(12, 10, str(row['student_id']), 1, 0, 'C', fill=True)
        pdf.cell(15, 10, str(row['risk_score']), 1, 0, 'C', fill=True)
        pdf.cell(12, 10, str(row['attendance']), 1, 0, 'C', fill=True)
        pdf.cell(10, 10, str(row['prev_gpa']), 1, 0, 'C', fill=True)
        
        # Clean text & combine explanation with Temporal Trajectory (Wider 55 char limit)
        explanation = str(row['risk_explanation']).encode('latin-1', 'ignore').decode('latin-1')
        trend = str(row['Trajectory']).encode('latin-1', 'ignore').decode('latin-1')
        full_reason = f"{explanation} [{trend}]"
        if len(full_reason) > 55: full_reason = full_reason[:52] + "..."
        pdf.cell(61, 10, full_reason, 1, 0, 'L', fill=True)
        
        # Clean allocation text
        allocation = str(row['allocation_status']).encode('latin-1', 'ignore').decode('latin-1')
        pdf.cell(80, 10, allocation, 1, 1, 'L', fill=True) 

    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ======================================================
# 1️⃣ APP CONFIGURATION & SIDEBAR
# ======================================================
st.set_page_config(page_title="EduTriage | Risk Optimizer", layout="wide", page_icon="🎓")

st.sidebar.title("⚙️ University Config")
st.sidebar.write("Customize rules and capacities.")

st.sidebar.divider()
st.sidebar.subheader("🚨 Strict Thresholds")
min_attendance = st.sidebar.slider("📅 Minimum Attendance Required (%)", 0, 100, 75)
min_gpa = st.sidebar.number_input("🎓 Minimum Passing GPA", 0.0, 10.0, 2.0, step=0.1)

st.sidebar.divider()
st.sidebar.subheader("⚖️ Risk Weighting")
w_att = st.sidebar.slider("📅 Weight: Attendance", 0, 100, 40)
w_gpa = st.sidebar.slider("🎓 Weight: GPA", 0, 100, 30)
w_marks = st.sidebar.slider("📝 Weight: Marks", 0, 100, 30)

st.sidebar.divider()
st.sidebar.subheader("🔒 Resource Constraints")
st.sidebar.info("Set available staff to trigger the Optimization Algorithm.")
max_counselors = st.sidebar.number_input("🧠 Max Counselors Available", 0, 50, 2)
max_tutors = st.sidebar.number_input("📚 Max Tutors Available", 0, 50, 1)

# ======================================================
# 2️⃣ MAIN APP LOGIC & BEAUTIFIED HEADER
# ======================================================

# 🚀 PROFESSIONAL CUSTOM HTML HEADER (UPDATED COLOR PALETTE)
st.markdown("""
    <div style='text-align: center; margin-top: -30px; padding-bottom: 20px;'>
        <h1 style='font-size: 3.5rem; color: #1A365D; margin-bottom: 0;'>🎓 EduTriage</h1>
        <h3 style='font-weight: 400; color: #2B6CB0; margin-top: 5px;'>Predictive Student Triage & Resource Optimizer</h3>
        <p style='color: #4A5568; font-style: italic; font-size: 1.1rem;'>A mathematical operations engine that allocates limited university resources to maximize student retention ROI.</p>
    </div>
    <hr style='border: 1px solid #EAECEE; margin-bottom: 30px;'>
""", unsafe_allow_html=True)

# --- SESSION STATE MEMORY FIX ---
# Teach the app to "remember" if we are using demo data
if "use_demo_data" not in st.session_state:
    st.session_state["use_demo_data"] = False

st.info("💡 Use the demo dataset to test the Optimizer!")
col_up1, col_up2 = st.columns([2, 1])

with col_up1:
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
with col_up2:
    st.write("") # Spacing
    st.write("")
    use_demo = st.button("🚀 Load Demo Data", use_container_width=True)

# If the button is clicked, turn the memory ON
if use_demo:
    st.session_state["use_demo_data"] = True

# Determine which data to load
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["use_demo_data"] = False # Turn off demo memory if a real file is uploaded
elif st.session_state["use_demo_data"]:
    # Read the demo data because the app remembers you wanted it!
    try:
        df = pd.read_csv("updated_students.csv") # Make sure this file is in your VS Code folder!
    except FileNotFoundError:
        try:
            df = pd.read_csv("students.csv") # Fallback just in case
        except FileNotFoundError:
            st.error("Demo file not found in repository.")

if df is not None:
    if "filter_status" not in st.session_state:
        st.session_state["filter_status"] = "High Risk"

    # TEMPORAL TRAJECTORY DATA GENERATOR
    if 'past_attendance' not in df.columns:
        np.random.seed(42)
        df['past_attendance'] = df['attendance'] - np.random.randint(-15, 20, len(df))
        df['past_attendance'] = df['past_attendance'].clip(0, 100)
    
    # Calculate Performance Trend (Numeric Math stays the same)
    df['att_velocity'] = df['attendance'] - df['past_attendance']
    
    # Create Visual UI Strings for Trajectory (Plain English, NO MINUS SIGNS)
    def format_trend(x):
        if x < 0:
            return f"📉 {abs(x)}% drop"
        elif x > 0:
            return f"📈 {x}% improved"
        else:
            return "➖ No change"
            
    df['Trajectory'] = df['att_velocity'].apply(format_trend)

    # ======================================================
    # 3️⃣ DYNAMIC RISK CALCULATION + TEMPORAL PENALTY
    # ======================================================
    risk_features = ["attendance", "ca_marks", "midterm_marks", "prev_gpa"]
    risk_scaler = MinMaxScaler()
    
    risk_scaler.fit(df[risk_features])
    norm_data = risk_scaler.transform(df[risk_features])
    norm_df = pd.DataFrame(norm_data, columns=risk_features)
    
    r_att = 1 - norm_df["attendance"]
    r_ca = 1 - norm_df["ca_marks"]
    r_mid = 1 - norm_df["midterm_marks"]
    r_gpa = 1 - norm_df["prev_gpa"]

    total_weight = w_att + w_gpa + w_marks + w_marks 
    base_weighted_score = ((r_att * w_att) + (r_gpa * w_gpa) + (r_ca * w_marks) + (r_mid * w_marks)) / total_weight
    base_risk = (base_weighted_score * 100)

    # APPLY TEMPORAL VELOCITY PENALTY
    temporal_penalty = df['att_velocity'].apply(lambda x: abs(x) * 0.8 if x < -5 else 0)
    df["risk_score"] = (base_risk + temporal_penalty).clip(0, 100).round(2)

    # ======================================================
    # 4️⃣ CLUSTERING
    # ======================================================
    features = df.drop(columns=["student_id", "risk_score", "past_attendance", "att_velocity", "Trajectory"], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_risk = df.groupby("cluster")["risk_score"].mean().sort_values(ascending=False).index
    df["cluster_level"] = df["cluster"].map({cluster_risk[0]: "High Risk", cluster_risk[1]: "Moderate Risk", cluster_risk[2]: "Safe"})

    # ======================================================
    # 5️⃣ "TRAFFIC CONTROLLER" & KNAPSACK OPTIMIZER
    # ======================================================
    def analyze_student(row):
        reasons = []
        if row['attendance'] < min_attendance: reasons.append(f"⚠️ Att < {min_attendance}%")
        if row['prev_gpa'] < min_gpa: reasons.append(f"⚠️ GPA < {min_gpa}")
        if row['att_velocity'] < -10: reasons.append("⚠️ Critical Drop") 
        
        if not reasons:
            if row['ca_marks'] < df['ca_marks'].quantile(0.25): reasons.append("Low CA")
            if row['midterm_marks'] < df['midterm_marks'].quantile(0.25): reasons.append("Low Midterm")
        
        explanation = ", ".join(reasons) if reasons else "Safe"

        # Base Ideal Intervention
        action = "None"
        if "High Risk" in row['cluster_level']:
            if ("Att" in explanation or "Drop" in explanation) and "GPA" in explanation: action = "🚨 Dean Meeting"
            elif "Att" in explanation or "Drop" in explanation: action = "🧠 Counselor"
            elif "GPA" in explanation: action = "📚 Tutor"
            else: action = "🔍 Review"
        elif "Moderate Risk" in row['cluster_level']: action = "🤝 Mentor"
        else: action = "✅ No Action"
            
        return pd.Series([explanation, action])

    df[["risk_explanation", "ideal_intervention"]] = df.apply(analyze_student, axis=1)

    # --- THE OPTIMIZATION ALGORITHM ---
    df["allocation_status"] = df["ideal_intervention"]

    # 1. Allocate Counselors by ROI
    c_mask = df['ideal_intervention'].str.contains("Counselor")
    c_df = df[c_mask].sort_values(by="risk_score", ascending=False)
    df.loc[c_df.head(max_counselors).index, "allocation_status"] = "🧠 Counselor (Allocated)"
    df.loc[c_df.iloc[max_counselors:].index, "allocation_status"] = "⏳ Waitlist (Counselor)"

    # 2. Allocate Tutors by ROI
    t_mask = df['ideal_intervention'].str.contains("Tutor")
    t_df = df[t_mask].sort_values(by="risk_score", ascending=False)
    df.loc[t_df.head(max_tutors).index, "allocation_status"] = "📚 Tutor (Allocated)"
    df.loc[t_df.iloc[max_tutors:].index, "allocation_status"] = "⏳ Waitlist (Tutor)"

    # ======================================================
    # 6️⃣ DASHBOARD UI
    # ======================================================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Students", len(df))
    c2.metric("⚠️ Violating Attendance", len(df[df['attendance'] < min_attendance]))
    c3.metric("📉 Violating GPA", len(df[df['prev_gpa'] < min_gpa]))
    
    nosedive_count = len(df[df['att_velocity'] <= -10])
    c4.metric("🚨 Critical Drops", nosedive_count, delta="-10% trend", delta_color="inverse")

    st.divider()

    # --- Resource Planning Board ---
    st.subheader("🛠️ Algorithmic Resource Allocation Board")
    
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    allocated_c = len(df[df['allocation_status'] == "🧠 Counselor (Allocated)"])
    waitlist_c = len(df[df['allocation_status'] == "⏳ Waitlist (Counselor)"])
    allocated_t = len(df[df['allocation_status'] == "📚 Tutor (Allocated)"])
    waitlist_t = len(df[df['allocation_status'] == "⏳ Waitlist (Tutor)"])

    col_r1.metric("🧠 Counselors Assigned", f"{allocated_c} / {max_counselors}")
    col_r2.metric("⏳ Counselor Waitlist", waitlist_c)
    col_r3.metric("📚 Tutors Assigned", f"{allocated_t} / {max_tutors}")
    col_r4.metric("⏳ Tutor Waitlist", waitlist_t)

    st.divider()

    # --- Circles & Filters ---
    col1, col2, col3, col4 = st.columns(4)
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
    with col4:
        pct_drop = round((nosedive_count / len(df)) * 100) if len(df) > 0 else 0
        st.markdown(f"<div style='text-align:center; color:#9c27b0; font-size:24px; font-weight:bold;'>{pct_drop}%</div>", unsafe_allow_html=True)
        if st.button("🚨 Critical Drops", use_container_width=True): st.session_state["filter_status"] = "Critical Drops"

    # --- Filtered Table ---
    st.subheader(f"📋 Student List: {st.session_state['filter_status']}")
    
    if st.session_state["filter_status"] == "Critical Drops":
        display_df = df[df['att_velocity'] <= -10].sort_values(by="risk_score", ascending=False)
    else:
        display_df = df[df["cluster_level"] == st.session_state["filter_status"]].sort_values(by="risk_score", ascending=False)

    def style_rows(row):
        color = '#ff4b4b' if "High" in row['cluster_level'] else '#ffa726' if "Moderate" in row['cluster_level'] else '#2e7d32'
        return [f'background-color: {color}80; color: white'] * len(row)

    # 🚀 BEAUTIFIED TABLE HEADERS WITH EMOJIS AND CUSTOM WIDTHS
    st.dataframe(
        display_df[["student_id", "risk_score", "attendance", "Trajectory", "risk_explanation", "allocation_status", "cluster_level"]]
        .style.apply(style_rows, axis=1),
        column_config={
            "student_id": st.column_config.TextColumn("🆔 Student ID", width="small"),
            "risk_score": st.column_config.NumberColumn("🎯 Risk Score %", format="%.2f"),
            "attendance": st.column_config.NumberColumn("📅 Att %"),
            "Trajectory": st.column_config.TextColumn("⏳ Performance Trend", width="medium"),
            "risk_explanation": st.column_config.TextColumn("📋 Risk Reason", width="large"),
            "allocation_status": st.column_config.TextColumn("🚀 Final Allocation", width="medium"),
            "cluster_level": st.column_config.TextColumn("🚥 Risk Level", width="small")
        },
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # 7️⃣ WHAT-IF SIMULATION
    # ======================================================
    st.divider()
    st.subheader("⚡ What-If Simulation")
    
    if not display_df.empty:
        s_id = st.selectbox("Select Student for Simulation:", display_df["student_id"].values)
        selected_student = df[df["student_id"] == s_id].iloc[0]

        col_sim1, col_sim2 = st.columns([1, 2])
        with col_sim1:
            st.info(f"**Current Risk:** {selected_student['risk_score']}")
            st.write(f"**Action:** {selected_student['allocation_status']}")
        
        with col_sim2:
            new_att = st.slider("📅 Attendance %", 0, 100, int(selected_student["attendance"]))
            new_gpa = st.slider("🎓 GPA", 0.0, 10.0, float(selected_student["prev_gpa"]))
            new_ca = st.slider("📝 CA Marks", 0, 100, int(selected_student["ca_marks"]))

            sim_row = pd.DataFrame([[new_att, new_ca, selected_student["midterm_marks"], new_gpa]], columns=risk_features)
            sim_norm = risk_scaler.transform(sim_row)
            
            s_r_att, s_r_ca = 1 - sim_norm[0][0], 1 - sim_norm[0][1]
            s_r_gpa = 1 - sim_norm[0][3]
            s_r_mid = 1 - risk_scaler.transform(pd.DataFrame([selected_student[risk_features]], columns=risk_features))[0][2]

            sim_weighted_score = ((s_r_att * w_att) + (s_r_gpa * w_gpa) + (s_r_ca * w_marks) + (s_r_mid * w_marks)) / total_weight
            new_risk = (sim_weighted_score * 100).round(2)

            st.metric("📊 Projected Base Risk", new_risk, delta=round(selected_student['risk_score'] - new_risk, 2))

    # ======================================================
    # 8️⃣ VISUALIZATION 
    # ======================================================
    st.divider()
    st.subheader("📈 Analysis Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = df["cluster_level"].map({"High Risk": "red", "Moderate Risk": "orange", "Safe": "green"})
    ax.scatter(df["attendance"], df["prev_gpa"], c=colors, alpha=0.5, label="Students")
    
    if not display_df.empty:
        ax.scatter(selected_student["attendance"], selected_student["prev_gpa"], c="black", s=200, marker="X", label=f"Selected ({selected_student['student_id']})")

    ax.axvline(x=min_attendance, color='blue', linestyle='--', label=f'Min Att ({min_attendance}%)')
    ax.axhline(y=min_gpa, color='purple', linestyle='--', label=f'Min GPA ({min_gpa})')
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("GPA")
    ax.legend()
    st.pyplot(fig)

    # ======================================================
    # 9️⃣ DOWNLOAD REPORTS (UPDATED WITH "CRITICAL DROPS")
    # ======================================================
    st.divider()
    st.subheader("📄 Download Reports")
    
    report_cat = st.selectbox("Select Category to Download:", ["All Students", "High Risk", "Moderate Risk", "Safe", "Critical Drops"])
    
    if st.button("🖨️ Generate PDF"):
        if report_cat == "All Students":
            r_df = df.sort_values(by="risk_score", ascending=False)
        elif report_cat == "Critical Drops":
            r_df = df[df['att_velocity'] <= -10].sort_values(by="risk_score", ascending=False)
        else:
            r_df = df[df["cluster_level"] == report_cat].sort_values(by="risk_score", ascending=False)
            
        if not r_df.empty:
            criteria_text = f"Att < {min_attendance}% | GPA < {min_gpa}"
            if report_cat == "Critical Drops":
                criteria_text += " | Trend <= -10%"
                
            pdf_data = create_category_pdf(r_df, report_cat, criteria_text)
            
            safe_filename = report_cat.replace(" ", "_")
            st.download_button("📥 Download PDF", pdf_data, f"{safe_filename}_Report.pdf", "application/pdf")
        else:
            st.warning("⚠️ No students in this category.")

# ======================================================
    # 10️⃣ AI EMAIL GENERATOR (GEMINI INTEGRATION)
    # ======================================================
    st.divider()
    st.subheader("✉️ Automated Empathy Outreach")
    
    if not display_df.empty:
        st.write(f"Drafting intervention email for **Student ID: {selected_student['student_id']}**")
        
        if st.button("✨ Generate AI Email Draft", type="primary"):
            with st.spinner("Analyzing risk profile and generating email..."):
                try:
                    # 1. Fetch the hidden API key from Streamlit Secrets
                    api_key = st.secrets["GEMINI_API_KEY"]
                    genai.configure(api_key=api_key)
                    
                    # 2. Use the fast, free-tier model
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # 3. Prompt Engineering: Injecting your math into the LLM
                    prompt = f"""
                    You are an empathetic, supportive Academic Dean at a university. 
                    Write a short, professional, but warm email to a student to check in on them.
                    
                    Here is their current academic data:
                    - Current Attendance: {selected_student['attendance']}%
                    - Recent Performance Trend: {selected_student['Trajectory']}
                    - Primary System Warning: {selected_student['risk_explanation']}
                    
                    Guidelines:
                    - Do not sound like a robot. Sound like a human who cares.
                    - Do not list their stats like a spreadsheet. Weave it into the conversation gently (e.g., "I noticed your attendance dipped recently...").
                    - End by inviting them to a low-pressure 10-minute chat with their assigned {selected_student['allocation_status'].split('(')[0].strip()}.
                    - Keep it under 3 paragraphs.
                    """
                    
                    # 4. Generate and display the email
                    response = model.generate_content(prompt)
                    st.text_area("Email Draft (Edit before sending):", response.text, height=300)
                    st.success("✅ Email drafted successfully based on temporal risk data.")
                    
                except Exception as e:
                    st.error("⚠️ Error: Please ensure your GEMINI_API_KEY is properly saved in Streamlit Cloud Secrets.")