
# JobMatch Lite V3.1

**A locally running CV and job description matching application built with Python and Streamlit.**

JobMatch Lite helps job seekers understand how their professional experience aligns with a job description through evidence-based requirement analysis.

## Features

- PDF CV text extraction.
- Job description requirement extraction.
- Skill and keyword matching.
- Direct, partial and transferable evidence classification.
- Identification of missing and unverified skills.
- Education and professional experience assessment.
- Work authorisation and preferred location profile.
- Weighted evidence, technical and domain scores.
- TF-IDF text similarity.
- Downloadable analysis report.

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- Scikit-learn
- TF-IDF and cosine similarity

## Privacy and Local Processing

The application performs its analysis locally.

- No paid APIs.
- No API keys.
- No external AI services.
- CV files are processed by the local application.

Do not upload personal CVs or confidential documents to a public repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/JobMatch-Lite.git
cd JobMatch-Lite
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```cmd
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app_v2.py
```

## Scoring

The evidence score considers direct, partial and transferable evidence. Example technologies are excluded from the score.

The technical and domain scores provide additional perspectives on the candidate's alignment with the role.

Text similarity is calculated using TF-IDF and cosine similarity.

**Important:** Scores are evidence-coverage indicators, not predictions of hiring outcomes. Final assessments depend on the employer.

## Current Version

V3.1 - Local evidence-based CV and job description analysis.

## Future Improvements

- Improved requirement classification.
- Better handling of alternative requirements.
- Enhanced domain-specific matching.
- Optional locally running language model.
- Application tracking.
- More detailed matching explanations.

## Author

Nithya Doresha

Data Analytics | Data Engineering | AI & Analytics

Open to opportunities across Sweden.
