"""
AI-Powered Resume Skill Analyzer - Flask REST API
---------------------------------------------------
Loads the TF-IDF model trained in Resume_Skill_Analyzer.ipynb and returns
job-fit scores, matched skills, missing skills, and top job recommendations
for a given resume text. No frontend, no database - just Flask + JSON.
"""

import re
import pickle
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------------------------------------------------
# Load the trained model artifacts (created by the notebook)
# ---------------------------------------------------------
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

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


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "AI Resume Skill Analyzer API Running"})


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
