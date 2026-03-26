import streamlit as st
from utils import (
    extract_text_from_pdf, clean_text, extract_keywords,
    get_similarity_score, get_bert_score, extract_skills,
    generate_suggestions, generate_pdf_report
)

# --- Page Config ---
st.set_page_config(
    page_title="Smart Recruiter",
    page_icon="🧠",
    layout="wide"
)

# --- Header ---
st.markdown("""
    <h1 style='text-align: center; color: #4A90D9;'>🧠 Smart Recruiter</h1>
    <p style='text-align: center; font-size: 18px; color: gray;'>
        AI-Powered Resume Analyzer — Upload your resume & get instant feedback
    </p>
    <hr>
""", unsafe_allow_html=True)

# --- Two Column Layout ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

with col_right:
    st.markdown("### 📋 Job Description")
    job_description = st.text_area("Paste the job description here", height=200)

st.markdown("---")

# --- Analyze Button ---
analyze = st.button("🚀 Analyze My Resume", use_container_width=True)

if analyze:
    if not uploaded_file:
        st.warning("Please upload your resume PDF first.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    else:
        with st.spinner("🔍 Analyzing your resume with AI..."):

            # Extract and clean
            resume_raw = extract_text_from_pdf(uploaded_file)
            resume_clean = clean_text(resume_raw)
            job_clean = clean_text(job_description)

            # Scores
            tfidf_score = get_similarity_score(resume_clean, job_clean)
            bert_score = get_bert_score(resume_clean, job_clean)
            combined = round((tfidf_score + bert_score) / 2, 2)

            # Skills
            resume_skills = set(extract_skills(resume_raw))
            job_skills = set(extract_skills(job_description))
            matched_skills = resume_skills & job_skills
            missing_skills = job_skills - resume_skills
            extra_skills = resume_skills - job_skills

            # Keywords
            resume_keywords = set(extract_keywords(resume_clean))
            job_keywords = set(extract_keywords(job_clean))
            matching_kw = resume_keywords & job_keywords
            missing_kw = job_keywords - resume_keywords

            # Suggestions
            suggestions = generate_suggestions(
                matched_skills, missing_skills, combined, missing_kw
            )

        # =====================
        # RESULTS UI
        # =====================

        # --- Score Cards ---
        st.markdown("## 📊 Match Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("TF-IDF Score", f"{tfidf_score}%", help="Keyword-based match")
        c2.metric("BERT Score", f"{bert_score}%", help="Meaning-based match")
        c3.metric("Combined Score", f"{combined}%", help="Overall match")

        # --- Progress Bar ---
        st.markdown("### Overall Match Progress")
        st.progress(int(combined))

        if combined >= 70:
            st.success("🟢 Strong Match — You're a great fit for this role!")
        elif combined >= 45:
            st.warning("🟡 Moderate Match — Tailor your resume a bit more.")
        else:
            st.error("🔴 Low Match — Significant changes needed.")

        st.markdown("---")

        # --- Skills ---
        st.markdown("## 🛠️ Skills Analysis")

        sk1, sk2 = st.columns(2)

        with sk1:
            st.markdown("**✅ Skills You Have**")
            if matched_skills:
                for s in sorted(matched_skills):
                    st.success(f"✔ {s}")
            else:
                st.info("No matching skills detected.")

        with sk2:
            st.markdown("**❌ Skills You're Missing**")
            if missing_skills:
                for s in sorted(missing_skills):
                    st.error(f"✘ {s}")
            else:
                st.success("You have all required skills! 🎉")

        if extra_skills:
            st.markdown("**💡 Bonus Skills on Your Resume**")
            st.info(", ".join(sorted(extra_skills)))

        st.markdown("---")

        # --- Keywords ---
        st.markdown("## 🔍 Keyword Analysis")

        kw1, kw2 = st.columns(2)
        with kw1:
            st.markdown("**✅ Matching Keywords**")
            st.success(", ".join(sorted(matching_kw)) if matching_kw else "None found")

        with kw2:
            st.markdown("**❌ Missing Keywords**")
            st.error(", ".join(sorted(missing_kw)) if missing_kw else "None missing!")

        st.markdown("---")

        # --- Suggestions ---
        st.markdown("## 💡 Personalized Improvement Suggestions")
        for i, s in enumerate(suggestions, 1):
            st.markdown(f"**{i}.** {s}")

        st.markdown("---")

        # --- Download Report ---
        st.markdown("## 📥 Download Your Report")
        pdf_bytes = generate_pdf_report(
            tfidf_score, bert_score, combined,
            matched_skills, missing_skills, suggestions
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="resume_analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.markdown("---")

        # --- Resume Preview ---
        with st.expander("📃 View Extracted Resume Text"):
            st.write(resume_raw)