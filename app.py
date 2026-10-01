"""
AI-Powered Resume Skill Analyzer - Flask REST API
---------------------------------------------------
Loads the TF-IDF model trained in Resume_Skill_Analyzer.ipynb and returns
job-fit scores, matched skills, missing skills, and top job recommendations
for a given resume text. No frontend, no database - just Flask + JSON.
"""

import re
import pickle
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from pypdf import PdfReader
from docx import Document
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------------------------------------------------
# Load the trained model artifacts (created by the notebook)
# ---------------------------------------------------------
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
job_market_df = pd.read_csv('job_descriptions.csv')
vectorizer = model_data['vectorizer']
job_vectors = model_data['job_vectors']
job_titles = model_data['job_titles']
SKILL_KEYWORDS = model_data['skill_keywords']
required_skills_map = model_data['required_skills_map']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------
# Text cleaning (same logic used in the notebook)
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_skills(text):
    found = []
    for skill in SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.append(skill)
    return found


def match_resume_to_jobs(cleaned_text, top_n=3):
    resume_vec = vectorizer.transform([cleaned_text])
    similarities = cosine_similarity(resume_vec, job_vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append({
            'title': job_titles[idx],
            'score': round(float(similarities[idx]) * 100, 2)
        })
    return results


def skill_gap_analysis(resume_skills, job_title):
    required = set(required_skills_map.get(job_title, []))
    have = set(resume_skills)
    matched = sorted(required & have)
    missing = sorted(required - have)
    return matched, missing

def extract_text_from_pdf(file):
    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_text_from_docx(file):
    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text
def get_market_skill_demand():

    skill_counts = {}

    for description in job_market_df['Job_Description']:

        skills = extract_skills(description)

        # Count a skill only once per job description
        for skill in set(skills):
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    total_jobs = len(job_market_df)

    market_data = []

    for skill, count in skill_counts.items():

        percentage = round((count / total_jobs) * 100, 1)

        market_data.append({
            'skill': skill,
            'count': count,
            'percentage': percentage
        })

    # Most demanded skills first
    market_data.sort(
        key=lambda x: x['count'],
        reverse=True
    )

    return market_data
@app.route('/job-market', methods=['GET'])
def job_market():

    market_data = get_market_skill_demand()

    return render_template(
        'job_market.html',
        market_data=market_data
    )
# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
#@app.route('/analyze', methods=['POST'])
def generate_learning_roadmap(missing_skills):
    roadmap = {
        "python": [
            "Python basics",
            "Functions and OOP",
            "NumPy and Pandas"
        ],
        "sql": [
            "SQL basics",
            "Joins and subqueries",
            "Advanced SQL"
        ],
        "pandas": [
            "Pandas Series and DataFrames",
            "Data cleaning",
            "Data analysis with Pandas"
        ],
        "statistics": [
            "Mean, median and variance",
            "Probability basics",
            "Hypothesis testing"
        ],
        "machine learning": [
            "Regression",
            "Classification",
            "Model evaluation"
        ],
        "numpy": [
            "NumPy arrays",
            "Array operations",
            "Linear algebra basics"
        ],
        "data visualization": [
            "Matplotlib",
            "Seaborn",
            "Dashboard creation"
        ]
    }

    result = {}

    for skill in missing_skills:
        if skill in roadmap:
            result[skill] = roadmap[skill]
        else:
            result[skill] = [
                f"Learn {skill} fundamentals",
                f"Practice {skill} with small projects",
                f"Apply {skill} in a real-world project"
            ]

    return result
@app.route('/analyze', methods=['POST'])
def analyze_resume():

    if 'resume' not in request.files:
        return jsonify({"error": "Please upload a resume."}), 400

    file = request.files['resume']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    filename = file.filename.lower()

    try:

        if filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(file)

        elif filename.endswith('.docx'):
            resume_text = extract_text_from_docx(file)

        else:
            return jsonify({
                "error": "Only PDF and DOCX files are supported."
            }), 400

        if not resume_text.strip():
            return jsonify({
                "error": "Could not extract text from the resume."
            }), 400

        # Clean resume text
        cleaned_text = clean_text(resume_text)

        # Extract skills
        resume_skills = extract_skills(cleaned_text)
                # Compare resume skills with job market demand
        market_data = get_market_skill_demand()

        resume_skill_set = set(resume_skills)

        market_skills = [item['skill'] for item in market_data[:10]]

        matched_market_skills = [
            skill for skill in market_skills
            if skill in resume_skill_set
        ]

        missing_market_skills = [
            skill for skill in market_skills
            if skill not in resume_skill_set
        ]

        if market_skills:
            market_coverage = round(
                (len(matched_market_skills) / len(market_skills)) * 100,
                1
            )
        else:
            market_coverage = 0
        learning_roadmap = generate_learning_roadmap(missing_market_skills)    

        # Find matching jobs
        top_matches = match_resume_to_jobs(
            cleaned_text,
            top_n=3
        )

        top_job = top_matches[0]['title']
        top_score = top_matches[0]['score']

        # Skill gap analysis
        matched_skills, missing_skills = skill_gap_analysis(
            resume_skills,
            top_job
        )

        return render_template(
           'results.html',
            top_job=top_job,
            match_score=f"{top_score}%",
            top_matches=top_matches,
            extracted_skills=resume_skills,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            market_data=market_data,
            market_coverage=market_coverage,
            learning_roadmap=learning_roadmap
)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    resume_text = data.get('resume_text', '')

    if not resume_text.strip():
        return jsonify({"error": "resume_text is required"}), 400

    cleaned_text = clean_text(resume_text)

    # Extract skills from the resume
    resume_skills = extract_skills(cleaned_text)

    # Get top 3 matching job roles
    top_matches = match_resume_to_jobs(cleaned_text, top_n=3)
    top_job = top_matches[0]['title']
    top_score = top_matches[0]['score']

    # Skill gap analysis against the top matching job
    matched_skills, missing_skills = skill_gap_analysis(resume_skills, top_job)

    response = {
        "Top Job": top_job,
        "Similarity Score": f"{top_score}%",
        "Top 3 Recommendations": [m['title'] for m in top_matches],
        "Matched Skills": matched_skills,
        "Missing Skills": missing_skills
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
