# 🧠 Smart Recruiter — AI Resume Analyzer

An AI-powered web application that analyzes resumes and compares them with job descriptions using NLP techniques like TF-IDF and BERT.

---

##  Features

*  Upload Resume (PDF)
*  Paste Job Description
*  TF-IDF Similarity Score
*  BERT Semantic Similarity
*  Skills Extraction & Matching
*  Keyword Analysis (Missing & Matching)
*  Personalized Suggestions
*  Downloadable PDF Report

---

## 🧰 Tech Stack

* Python
* Streamlit
* NLTK
* Scikit-learn
* Sentence Transformers (BERT)
* PyPDF2
* FPDF

---

## 📂 Project Structure

smart-recruiter/
│── app.py
│── utils.py
│── requirements.txt
│── README.md

---

## ⚙️ Installation & Run Locally

```bash
# Clone repo
git clone https://github.com/your-username/smart-recruiter.git

# Go to folder
cd smart-recruiter

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 🌐 Live Demo

👉 (Add your Streamlit link here after deployment)

---

## 📊 How It Works

1. Extracts text from resume PDF
2. Cleans & processes text using NLP
3. Calculates:

   * TF-IDF similarity (keyword match)
   * BERT similarity (semantic meaning)
4. Identifies:

   * Matching skills
   * Missing skills
   * Keyword gaps
5. Generates improvement suggestions
6. Creates downloadable PDF report

---

## 💡 Future Improvements

* Multi-resume ranking system
* Job recommendation engine
* User login & dashboard
* Database integration

---

## 🙌 Author

Neha
(You can add your LinkedIn / GitHub here)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
