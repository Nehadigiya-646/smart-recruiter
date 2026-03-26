import PyPDF2
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data (only runs once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# ── STEP 2: Extract text from uploaded PDF ──────────────────────────────────

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    return text.strip()


# ── STEP 3: Clean the text (NLP preprocessing) ─────────────────────────────

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if not word.isdigit()]
    return " ".join(tokens)


# ── HELPER: Extract keywords from text ─────────────────────────────────────

def extract_keywords(text, top_n=15):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    freq = {}
    for word in tokens:
        if len(word) > 2:
            freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:top_n]]


# ── STEP 3: Calculate similarity score ─────────────────────────────────────

def get_similarity_score(resume_text, job_text):
    """
    TF-IDF converts text into numbers based on word importance.
    Cosine Similarity measures how close two texts are (0 to 100).
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_job    = clean_text(job_text)

    vectorizer = TfidfVectorizer()
    vectors    = vectorizer.fit_transform([cleaned_resume, cleaned_job])
    score      = cosine_similarity(vectors[0], vectors[1])[0][0]

    return round(score * 100, 2)


# ── STEP 3: Find matching and missing keywords ──────────────────────────────

def get_matching_keywords(resume_text, job_text):
    """
    matched = words in BOTH resume and job (good!)
    missing = words in job but NOT in resume (gaps to fix)
    """
    resume_keywords = set(extract_keywords(resume_text, top_n=40))
    job_keywords    = set(extract_keywords(job_text,    top_n=40))

    matched = list(resume_keywords & job_keywords)
    missing = list(job_keywords - resume_keywords)

    return matched, missing


# ── STEP 3: Generate feedback based on score ───────────────────────────────

def get_feedback(score):
    if score >= 70:
        return "strong", "Great match! Your resume aligns well with this job description."
    elif score >= 45:
        return "medium", "Decent match. Consider adding some missing keywords to your resume."
    else:
        return "weak", "Low match. Your resume may need significant updates for this role."

from sentence_transformers import SentenceTransformer, util

# Load model once (outside the function so it doesn't reload every time)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_bert_score(resume_text, job_text):
    """
    Uses BERT to compute semantic similarity between resume and job description.
    Returns a score between 0 and 100.
    """
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = bert_model.encode(job_text, convert_to_tensor=True)
    
    score = util.cos_sim(resume_embedding, job_embedding)
    return round(float(score[0][0]) * 100, 2)

# Master skills database
SKILLS_DB = [
    # Programming Languages
    "python", "java", "javascript", "c++", "c#", "r", "sql", "php", "swift", "kotlin",
    "typescript", "ruby", "scala", "go", "rust", "matlab",

    # ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "neural network", "bert", "transformers", "gpt",
    "reinforcement learning", "data science",

    # Libraries & Frameworks
    "pandas", "numpy", "scikit-learn", "tensorflow", "keras", "pytorch",
    "streamlit", "flask", "django", "fastapi", "opencv", "spacy", "nltk",
    "huggingface", "langchain",

    # Databases
    "mysql", "postgresql", "mongodb", "sqlite", "redis", "firebase",
    "elasticsearch", "cassandra",

    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
    "ci/cd", "linux", "jenkins",

    # Data & BI Tools
    "tableau", "power bi", "excel", "hadoop", "spark", "airflow",

    # Soft Skills
    "communication", "teamwork", "leadership", "problem solving",
    "time management", "critical thinking", "adaptability"
]

def extract_skills(text):
    """
    Scans resume/job text and returns all matched skills from the database.
    """
    text_lower = text.lower()
    found_skills = []
    for skill in SKILLS_DB:
        if skill in text_lower:
            found_skills.append(skill)
    return list(set(found_skills))  # remove duplicates

def generate_suggestions(matched_skills, missing_skills, combined_score, missing_keywords):
    """
    Generates personalized improvement suggestions based on gap analysis.
    """
    suggestions = []

    # Score-based general advice
    if combined_score < 45:
        suggestions.append("🔴 Your resume needs major tailoring for this role. Focus on the missing skills below.")
    elif combined_score < 70:
        suggestions.append("🟡 You're a moderate match. A few targeted additions could make you competitive.")
    else:
        suggestions.append("🟢 Strong match! Just fine-tune a few areas to stand out even more.")

    # Missing skills advice
    if missing_skills:
        skills_list = ", ".join(sorted(missing_skills))
        suggestions.append(f"📌 Add these missing skills to your resume if you know them: **{skills_list}**")
        suggestions.append("📚 If you don't know these skills yet, consider doing a short online course (Coursera, YouTube) and adding a small project.")

    # Missing keywords advice
    if missing_keywords:
        kw_list = ", ".join(sorted(list(missing_keywords)[:8]))  # show top 8
        suggestions.append(f"🔑 Use these keywords naturally in your resume: **{kw_list}**")

    # Matched skills encouragement
    if matched_skills:
        suggestions.append(f"✅ You already have {len(matched_skills)} matching skill(s) — make sure they are clearly visible and prominent on your resume.")

    # General tips
    suggestions.append("📝 Tailor your resume summary/objective section to mirror the job description language.")
    suggestions.append("📊 Add measurable achievements — e.g. 'Built NLP model with 89% accuracy' instead of just 'Built NLP model'.")
    suggestions.append("🔗 Add a GitHub profile link with relevant projects to strengthen your application.")

    return suggestions

from fpdf import FPDF

def generate_pdf_report(tfidf_score, bert_score, combined_score,
                         matched_skills, missing_skills, suggestions):
    """
    Generates a downloadable PDF report of the resume analysis.
    """
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 12, "Smart Recruiter - Resume Analysis Report", ln=True, align="C")
    pdf.ln(5)

    # Scores
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Match Scores", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"  TF-IDF Score  : {tfidf_score}%", ln=True)
    pdf.cell(0, 8, f"  BERT Score    : {bert_score}%", ln=True)
    pdf.cell(0, 8, f"  Combined Score: {combined_score}%", ln=True)
    pdf.ln(4)

    # Matched Skills
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Matched Skills", ln=True)
    pdf.set_font("Arial", "", 12)
    if matched_skills:
        pdf.multi_cell(0, 8, "  " + ", ".join(sorted(matched_skills)))
    else:
        pdf.cell(0, 8, "  None found.", ln=True)
    pdf.ln(4)

    # Missing Skills
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Missing Skills", ln=True)
    pdf.set_font("Arial", "", 12)
    if missing_skills:
        pdf.multi_cell(0, 8, "  " + ", ".join(sorted(missing_skills)))
    else:
        pdf.cell(0, 8, "  None! Great match.", ln=True)
    pdf.ln(4)

    # Suggestions
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Improvement Suggestions", ln=True)
    pdf.set_font("Arial", "", 11)
    for i, s in enumerate(suggestions, 1):
        # Remove emojis for PDF compatibility
        clean_s = s.encode("latin-1", "ignore").decode("latin-1")
        pdf.multi_cell(0, 8, f"  {i}. {clean_s}")
        pdf.ln(1)

    return bytes(pdf.output())