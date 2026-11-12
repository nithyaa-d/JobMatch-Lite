

import re
import streamlit as st
import pymupdf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# JOBMATCH LITE V3.1
# Local, evidence-based CV and job description analysis
# ============================================================

st.set_page_config(
    page_title="JobMatch Lite V3.1",
    page_icon="??",
    layout="wide"
)

st.title("?? JobMatch Lite V3.1")
st.caption(
    "Local CV intelligence | Evidence-based matching | "
    "Requirement classification"
)

st.info(
    "All analysis runs locally. No paid APIs, API keys "
    "or external AI services are used."
)


# ============================================================
# 1. TEXT PROCESSING
# ============================================================

def normalize(text):
    if not text:
        return ""

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\u2022": "",
        "\x00": " "
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_line(line):
    return re.sub(
        r"^[\s???\---]+",
        "",
        normalize(line)
    ).strip()


def split_lines(text):
    output = []

    for raw in normalize(text).split("\n"):
        line = clean_line(raw)

        if not line:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            line
        )

        for sentence in sentences:
            sentence = clean_line(sentence)

            if len(sentence) >= 5:
                output.append(sentence)

    return output


def extract_pdf(uploaded_file):
    doc = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    pages = []

    for page in doc:
        pages.append(page.get_text("text", sort=True))

    doc.close()

    return normalize("\n".join(pages))


# ============================================================
# 2. SKILL TAXONOMY
# ============================================================

SKILLS = {
    "Programming": {
        "Python": ["python"],
        "SQL": ["sql", "structured query language"],
        "R": ["r programming", "r language"],
        "Java": ["java"],
        "C++": ["c++"]
    },

    "Data Engineering": {
        "ETL / ELT": ["etl", "elt"],
        "Data Pipelines": ["data pipelines", "data pipeline"],
        "Data Modelling": ["data modelling", "data modeling"],
        "Data Warehousing": ["data warehousing", "data warehouse"],
        "Data Quality": ["data quality", "data validation"],
        "Apache Spark": ["apache spark", "pyspark", "spark"],
        "Databricks": ["databricks"],
        "Snowflake": ["snowflake"],
        "BigQuery": ["bigquery", "big query"],
        "dbt": ["dbt"],
        "Git": [
            "git version control", "git workflow",
            "git repository", "git branching"
        ],
        "DevOps": [
            "devops", "continuous integration",
            "continuous deployment", "ci/cd"
        ],
        "APIs": ["api integration", "rest api", "apis"]
    },

    "Cloud": {
        "AWS": ["amazon web services", "aws"],
        "Azure": ["microsoft azure", "azure"],
        "Google Cloud": ["google cloud", "gcp"],
        "SageMaker": ["sagemaker"],
        "Bedrock": ["aws bedrock", "amazon bedrock"]
    },

    "Analytics & Statistics": {
        "Data Analysis": ["data analysis", "data analytics"],
        "Statistical Analysis": [
            "statistical analysis", "statistical methods",
            "statistics"
        ],
        "Hypothesis Testing": ["hypothesis testing"],
        "A/B Testing": ["a/b testing", "ab testing"],
        "Data Visualisation": [
            "data visualisation", "data visualization"
        ],
        "Power BI": ["power bi"],
        "Tableau": ["tableau"],
        "Excel": [
            "microsoft excel", "ms excel",
            "advanced excel", "excel spreadsheet"
        ],
        "Forecasting": ["forecasting", "forecast models"],
        "Automation": ["automation", "automated reporting"],
        "Quality Control": [
            "quality control", "quality assurance"
        ]
    },

    "Machine Learning & AI": {
        "Machine Learning": ["machine learning"],
        "Deep Learning": ["deep learning"],
        "NLP": ["natural language processing", "nlp"],
        "Generative AI": ["generative ai", "genai"],
        "LLMs": ["large language models", "llms", "llm"],
        "Predictive Modelling": [
            "predictive modelling", "predictive modeling"
        ],
        "Feature Engineering": ["feature engineering"],
        "Scikit-learn": ["scikit-learn", "sklearn"],
        "TensorFlow": ["tensorflow"],
        "PyTorch": ["pytorch"],
        "SHAP": ["shap"]
    },

    "Specialist Domains": {
        "Bioinformatics": ["bioinformatics"],
        "Biostatistics": ["biostatistics"],
        "Proteomics": ["proteomics"],
        "Genomics": ["genomics"],
        "Biotechnology": ["biotechnology", "biotech"],
        "Biopharma": ["biopharma", "biopharmaceutical"],
        "Life Sciences": ["life sciences"],
        "Healthcare": ["healthcare", "health care"],
        "Clinical Research": ["clinical research"],
        "Diagnostics": ["diagnostics", "diagnostic"],
        "Molecular Biology": ["molecular biology"],
        "Laboratory Data": ["laboratory data", "lab data"],
        "Drug Discovery": ["drug discovery"],
        "Biomarker Discovery": ["biomarker discovery"]
    }
}

DOMAIN_CATEGORY = "Specialist Domains"

WEIGHTS = {
    "Must-have": 3.0,
    "Preferred": 1.5,
    "Mentioned": 1.0
}

TRANSFERABLE = {
    "Bioinformatics": [
        "Python", "SQL", "Data Analysis",
        "Statistical Analysis", "Machine Learning"
    ],
    "Biostatistics": [
        "Statistical Analysis", "Hypothesis Testing",
        "Data Analysis", "Python"
    ],
    "Proteomics": [
        "Data Analysis", "Python", "SQL",
        "Statistical Analysis", "Machine Learning"
    ],
    "Genomics": [
        "Python", "SQL", "Data Analysis",
        "Statistical Analysis", "Machine Learning"
    ],
    "Biotechnology": [
        "Data Analysis", "Statistical Analysis",
        "Python", "Machine Learning"
    ],
    "Biopharma": [
        "Data Analysis", "Forecasting",
        "Statistical Analysis", "Python"
    ],
    "Life Sciences": [
        "Data Analysis", "Statistical Analysis",
        "Python", "Machine Learning"
    ],
    "Healthcare": [
        "Data Analysis", "Statistical Analysis",
        "Python", "Quality Control"
    ],
    "Clinical Research": [
        "Statistical Analysis", "Data Analysis",
        "Hypothesis Testing", "Quality Control"
    ],
    "Diagnostics": [
        "Data Analysis", "Statistical Analysis",
        "Quality Control", "Python"
    ],
    "Molecular Biology": [
        "Python", "Data Analysis", "Statistics"
    ],
    "Drug Discovery": [
        "Machine Learning", "Python",
        "Statistical Analysis", "Data Analysis"
    ],
    "Biomarker Discovery": [
        "Statistical Analysis", "Machine Learning",
        "Python", "Data Analysis"
    ]
}


# ============================================================
# 3. SKILL MATCHING
# ============================================================

def contains_phrase(text, phrase):
    text = normalize(text)

    if phrase == "R":
        return bool(re.search(
            r"(?<![A-Za-z0-9])R(?![A-Za-z0-9&])",
            text
        ))

    pattern = (
        r"(?<![A-Za-z0-9])"
        + re.escape(phrase)
        + r"(?![A-Za-z0-9])"
    )

    return bool(re.search(pattern, text, re.IGNORECASE))


def aliases_for(skill):
    for category in SKILLS.values():
        if skill in category:
            return category[skill]

    return []


def skill_found(text, skill, aliases):
    text = normalize(text)

    if skill == "R":
        explicit = any(
            contains_phrase(text, alias)
            for alias in aliases
        )

        standalone = contains_phrase(text, "R")

        context = bool(re.search(
            r"\b(programming|language|proficiency|"
            r"statistical|coding|data processing)\b",
            text,
            re.IGNORECASE
        ))

        return explicit or (standalone and context)

    return any(
        contains_phrase(text, alias)
        for alias in aliases
    )


def detect_cv_skills(text):
    found = {}

    for category, skills in SKILLS.items():
        for skill, aliases in skills.items():
            if skill_found(text, skill, aliases):
                found[skill] = category

    return found


# ============================================================
# 4. CV SECTIONS AND EVIDENCE
# ============================================================

CV_HEADINGS = {
    "skills": [
        "skills", "technical skills", "technical expertise",
        "tools and technologies", "programming skills",
        "cloud and platforms"
    ],
    "experience": [
        "professional experience", "work experience",
        "employment history", "experience"
    ],
    "projects": [
        "projects", "selected projects", "key projects"
    ],
    "education": [
        "education", "academic background", "qualifications"
    ]
}


def identify_section(line, current):
    lower = line.lower().strip().rstrip(":")

    for section, headings in CV_HEADINGS.items():
        if lower in headings:
            return section

    return current


def cv_records(text):
    records = []
    current = "other"

    for raw in normalize(text).split("\n"):
        line = clean_line(raw)

        if not line:
            continue

        new_section = identify_section(line, current)

        if new_section != current:
            current = new_section
            continue

        records.append({
            "text": line,
            "section": current
        })

    return records


def evidence_type(record):
    section = record["section"]
    text = record["text"].lower()

    action_words = [
        "built", "developed", "implemented", "designed",
        "created", "automated", "integrated", "deployed",
        "improved", "reduced", "increased", "delivered",
        "forecast", "analysed", "analyzed", "managed",
        "migrated", "optimised", "optimized", "trained",
        "tested", "led", "maintained"
    ]

    if section in ["experience", "projects"]:
        if any(word in text for word in action_words):
            return "Project / work evidence"

        return "Employment / project mention"

    if section == "skills":
        return "Skills list evidence"

    if section == "education":
        return "Education evidence"

    return "Other CV mention"


def get_evidence(cv_text, skill):
    records = cv_records(cv_text)
    aliases = aliases_for(skill)
    results = []

    for record in records:
        if skill_found(record["text"], skill, aliases):
            results.append({
                "type": evidence_type(record),
                "text": record["text"][:450]
            })

    priority = {
        "Project / work evidence": 0,
        "Employment / project mention": 1,
        "Skills list evidence": 2,
        "Education evidence": 3,
        "Other CV mention": 4
    }

    results.sort(
        key=lambda item: priority.get(item["type"], 5)
    )

    return results[:4]


# ============================================================
# 5. JOB DESCRIPTION PRIORITY
# ============================================================

MUST_HEADINGS = [
    "requirements", "required qualifications",
    "minimum qualifications", "essential requirements",
    "essential criteria", "what you need",
    "what you'll need", "what you will need",
    "your qualifications", "must have", "required skills"
]

PREFERRED_HEADINGS = [
    "preferred qualifications", "preferred skills",
    "nice to have", "nice-to-have",
    "bonus qualifications", "additional qualifications",
    "desirable skills"
]

RESET_PREFIXES = [
    "about the job", "about us",
    "location/division specific information",
    "a day in the life", "responsibilities",
    "what you will do", "what you'll do",
    "what we offer", "why join us",
    "to succeed in this role",
    "to be successful in this role",
    "about you", "in this role"
]


def heading_priority(line):
    lower = clean_line(line).lower().rstrip(": ")

    if lower in MUST_HEADINGS:
        return "Must-have"

    if lower in PREFERRED_HEADINGS:
        return "Preferred"

    return None


def is_reset_section(line):
    lower = clean_line(line).lower()

    return any(
        lower.startswith(prefix)
        for prefix in RESET_PREFIXES
    )


def explicit_priority(line):
    lower = line.lower()

    if any(x in lower for x in [
        "preferred", "nice to have", "nice-to-have",
        "bonus", "ideally", "desirable"
    ]):
        return "Preferred"

    if any(x in lower for x in [
        "mandatory", "essential", "must-have",
        "must have", "we require"
    ]):
        return "Must-have"

    return None


# ============================================================
# 6. EXAMPLE TECHNOLOGY DETECTION
# ============================================================

def is_example_technology(skill, line):
    lower = line.lower()

    # Examples commonly introduced by these expressions.
    example_markers = [
        "e.g.", "e.g,", "for example",
        "such as", "including", "like"
    ]

    positions = [
        lower.find(marker)
        for marker in example_markers
        if lower.find(marker) >= 0
    ]

    if not positions:
        return False

    aliases = aliases_for(skill)

    for alias in aliases:
        position = lower.find(alias.lower())

        if position >= 0 and any(
            marker_position < position
            for marker_position in positions
        ):
            return True

    return False


# ============================================================
# 7. EDUCATION REQUIREMENTS
# ============================================================

EDUCATION_SUBJECTS = [
    "Bioinformatics", "Biostatistics",
    "Computer Science", "Molecular Biology",
    "Biotechnology", "Life Sciences",
    "Statistics", "Mathematics", "Engineering"
]


def degree_requirement_line(line):
    degree = re.search(
        r"\b(msc|m\.sc|phd|master'?s|bachelor'?s|"
        r"b\.tech|bsc|b\.sc|advanced degree)\b",
        line,
        re.IGNORECASE
    )

    context = re.search(
        r"\b(in|degree|qualification|similar|related field)\b",
        line,
        re.IGNORECASE
    )

    return bool(degree and context)


def extract_education_subjects(job_text):
    found = []

    for line in split_lines(job_text):
        if not degree_requirement_line(line):
            continue

        for subject in EDUCATION_SUBJECTS:
            if contains_phrase(line, subject):
                if subject not in found:
                    found.append(subject)

    return found


def extract_degree_levels(text):
    patterns = {
        "PhD": r"\b(ph\.?\s*d|doctorate|doctoral)\b",
        "Master's": r"\b(master'?s|msc|m\.sc|m\.tech)\b",
        "Bachelor's": (
            r"\b(bachelor'?s|b\.?\s*tech|b\.?\s*e\.|"
            r"bsc|b\.sc|undergraduate)\b"
        )
    }

    return [
        name for name, pattern in patterns.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]


# ============================================================
# 8. REQUIREMENT EXTRACTION
# ============================================================

def extract_requirements(job_text):
    requirements = {}
    current_priority = "Mentioned"

    for line in split_lines(job_text):

        heading = heading_priority(line)

        if heading:
            current_priority = heading
            continue

        if is_reset_section(line):
            current_priority = "Mentioned"

        priority = explicit_priority(line) or current_priority

        # Degree subjects are assessed separately.
        education_line = degree_requirement_line(line)

        for category, skills in SKILLS.items():
            for skill, aliases in skills.items():

                if (
                    education_line
                    and category == DOMAIN_CATEGORY
                ):
                    continue

                if not skill_found(line, skill, aliases):
                    continue

                example = is_example_technology(skill, line)

                if skill not in requirements:
                    requirements[skill] = {
                        "category": category,
                        "priority": priority,
                        "example": example,
                        "evidence": [line]
                    }
                else:
                    item = requirements[skill]

                    if WEIGHTS[priority] > WEIGHTS[item["priority"]]:
                        item["priority"] = priority

                    item["example"] = (
                        item["example"] and example
                    )

                    if line not in item["evidence"]:
                        item["evidence"].append(line)

    return requirements


# ============================================================
# 9. DOMAIN ALTERNATIVES
# ============================================================

def extract_domain_groups(job_text):
    groups = []

    for line in split_lines(job_text):
        matches = []

        for skill in SKILLS[DOMAIN_CATEGORY]:
            for alias in aliases_for(skill):
                pattern = re.compile(
                    r"(?<![A-Za-z0-9])"
                    + re.escape(alias)
                    + r"(?![A-Za-z0-9])",
                    re.IGNORECASE
                )

                match = pattern.search(line)

                if match:
                    matches.append({
                        "skill": skill,
                        "start": match.start(),
                        "end": match.end()
                    })
                    break

        matches.sort(key=lambda x: x["start"])

        # An OR group must have an actual "or" between
        # domain terms, not just somewhere in the sentence.
        or_positions = [
            match.start()
            for match in re.finditer(
                r"\bor\b", line, re.IGNORECASE
            )
        ]

        for or_position in or_positions:
            left = [
                item for item in matches
                if item["end"] <= or_position
            ]

            right = [
                item for item in matches
                if item["start"] > or_position
            ]

            if not left or not right:
                continue

            left_item = left[-1]
            right_item = right[0]

            if left_item["skill"] == right_item["skill"]:
                continue

            # Include the coordinated domain list only when
            # it appears in an experience/environment context.
            lower = line.lower()

            context = any(term in lower for term in [
                "experience", "environment", "sector",
                "industry", "integrating", "harmonizing",
                "analysing", "analyzing"
            ])

            if not context:
                continue

            names = []

            for item in matches:
                if item["skill"] not in names:
                    names.append(item["skill"])

            if len(names) >= 2:
                groups.append({
                    "name": "Relevant domain experience",
                    "alternatives": names,
                    "priority": "Must-have",
                    "evidence": line
                })

            break

    return groups


def grouped_domains(groups):
    result = set()

    for group in groups:
        result.update(group["alternatives"])

    return result


# ============================================================
# 10. REQUIREMENT CLASSIFICATION
# ============================================================

def classify_requirement(skill, requirement, cv_skills, cv_text):
    if requirement["example"]:
        return "Example technology"

    if skill in cv_skills:
        evidence = get_evidence(cv_text, skill)

        if any(
            item["type"] == "Project / work evidence"
            for item in evidence
        ):
            return "Direct match"

        if any(
            item["type"] in [
                "Employment / project mention",
                "Skills list evidence"
            ]
            for item in evidence
        ):
            return "Partial match"

    if skill in TRANSFERABLE:
        related = [
            item for item in TRANSFERABLE[skill]
            if item in cv_skills
        ]

        if related:
            return "Transferable"

    if requirement["category"] == DOMAIN_CATEGORY:
        return "Domain gap"

    return "Not evidenced"


# ============================================================
# 11. EXPERIENCE PATHWAYS
# ============================================================

def extract_experience_pathways(job_text):
    pathways = []

    for line in split_lines(job_text):
        if not re.search(r"\b(years?|yrs?)\b", line, re.IGNORECASE):
            continue

        clauses = re.split(
            r"\s+\bor\b\s+",
            line,
            flags=re.IGNORECASE
        )

        for clause in clauses:
            match = re.search(
                r"(\d+)\+?\s*(?:years?|yrs?)",
                clause,
                re.IGNORECASE
            )

            if not match:
                continue

            years = int(match.group(1))
            lower = clause.lower()

            if "advanced degree" in lower:
                pathways.append({
                    "degree": "Advanced degree",
                    "years": years,
                    "source": clause.strip()
                })
                continue

            degree_patterns = [
                ("PhD", r"\b(ph\.?\s*d|doctorate|doctoral)\b"),
                ("Master's", r"\b(master'?s|msc|m\.sc|m\.tech)\b"),
                ("Bachelor's", r"\b(bachelor'?s|b\.?\s*tech|bsc|b\.sc)\b")
            ]

            for degree, pattern in degree_patterns:
                if re.search(pattern, lower):
                    pathways.append({
                        "degree": degree,
                        "years": years,
                        "source": clause.strip()
                    })

    unique = {}

    for item in pathways:
        unique[(item["degree"], item["years"])] = item

    return list(unique.values())


def pathway_matches(candidate_degree, required_degree):
    if required_degree == "Advanced degree":
        return candidate_degree in ["Master's", "PhD"]

    if required_degree == "Master's":
        return candidate_degree in ["Master's", "PhD"]

    if required_degree == "PhD":
        return candidate_degree == "PhD"

    if required_degree == "Bachelor's":
        return candidate_degree in [
            "Bachelor's", "Master's", "PhD"
        ]

    return False


def assess_experience(pathways, candidate_degree, years):
    applicable = [
        item for item in pathways
        if pathway_matches(candidate_degree, item["degree"])
    ]

    if not applicable:
        return {
            "status": "Manual review",
            "message": (
                "No directly applicable pathway was detected. "
                "Check the qualification and experience requirement."
            )
        }

    best = min(applicable, key=lambda x: x["years"])
    difference = round(best["years"] - years, 1)

    if difference <= 0:
        return {
            "status": "Meets detected experience threshold",
            "message": (
                f"Your declared professional experience meets "
                f"the {best['degree']} pathway of {best['years']} years."
            )
        }

    return {
        "status": "Below detected experience threshold",
        "message": (
            f"Your declared professional experience is "
            f"{difference} years below the {best['degree']} "
            f"pathway of {best['years']} years."
        )
    }


# ============================================================
# 12. CONTEXTUAL REQUIREMENTS
# ============================================================

def extract_contextual_requirements(job_text):
    contexts = []

    patterns = [
        (
            "Medical or biological research statistics",
            r"statistical methods?.{0,100}"
            r"(medical|biological|clinical).{0,120}"
            r"(research|data)"
        ),
        (
            "High-throughput technology experience",
            r"high[- ]throughput.{0,100}"
            r"(proteomics|genomics)"
        ),
        (
            "Scientific domain experience",
            r"(scientific|medical|biological).{0,80}"
            r"(research|data|analysis)"
        )
    ]

    for label, pattern in patterns:
        match = re.search(
            pattern,
            job_text,
            re.IGNORECASE
        )

        if match:
            contexts.append({
                "label": label,
                "evidence": match.group(0)
            })

    return contexts


def assess_contextual_requirement(label, cv_text):
    lower = cv_text.lower()

    if "medical or biological research statistics" in label.lower():
        terms = [
            "medical research", "biological research",
            "clinical research", "biostatistics"
        ]

        if any(term in lower for term in terms):
            return "Direct evidence detected"

        if any(term in lower for term in [
            "statistical analysis", "hypothesis testing",
            "statistical modelling", "statistical modeling"
        ]):
            return "Partial / transferable evidence"

        return "Not evidenced"

    if "high-throughput" in label.lower():
        if any(term in lower for term in [
            "high-throughput", "proteomics", "genomics"
        ]):
            return "Direct evidence detected"

        return "Not evidenced"

    if "scientific domain" in label.lower():
        if any(term in lower for term in [
            "scientific research", "biological research",
            "medical research", "clinical research"
        ]):
            return "Direct evidence detected"

        return "Not evidenced"

    return "Manual review"


# ============================================================
# 13. SCORING
# ============================================================

def calculate_requirement_score(requirements, classifications):
    total = 0.0
    earned = 0.0

    for skill, item in requirements.items():
        classification = classifications[skill]

        if classification == "Example technology":
            continue

        weight = WEIGHTS[item["priority"]]
        total += weight

        if classification == "Direct match":
            earned += weight

        elif classification == "Partial match":
            earned += weight * 0.6

        elif classification == "Transferable":
            earned += weight * 0.35

    if total == 0:
        return None

    return round(earned / total * 100, 1)


def category_score(requirements, classifications, category):
    subset = {
        skill: item
        for skill, item in requirements.items()
        if item["category"] == category
    }

    return calculate_requirement_score(
        subset,
        classifications
    )


def text_similarity(cv_text, job_text):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=6000
        )

        matrix = vectorizer.fit_transform([
            cv_text,
            job_text
        ])

        return round(
            cosine_similarity(
                matrix[0:1],
                matrix[1:2]
            )[0][0] * 100,
            1
        )

    except ValueError:
        return 0.0


# ============================================================
# 14. REPORT
# ============================================================

def create_report(
    profile,
    overall,
    technical,
    domain,
    similarity,
    classifications,
    requirements,
    pathways,
    experience_result,
    education_subjects,
    contextual
):
    lines = [
        "JOBMATCH LITE V3.1",
        "=" * 55,
        "",
        "CANDIDATE PROFILE",
        f"Degree: {profile['degree']}",
        f"Degree subject: {profile['subject']}",
        f"Degree duration: {profile['duration']} years",
        f"Professional experience: {profile['experience']} years",
        f"Work authorisation: {profile['authorisation']}",
        f"Preferred location: {profile['location']}",
        "",
        "MATCH SCORES",
        f"Requirement score: {overall}%",
        f"Technical score: {technical}%",
        f"Specialist domain score: {domain}%",
        f"Text similarity: {similarity}%",
        "",
        "REQUIREMENT CLASSIFICATIONS"
    ]

    for category in [
        "Direct match", "Partial match", "Transferable",
        "Domain gap", "Not evidenced", "Example technology"
    ]:
        lines.append("")
        lines.append(category)

        for skill, classification in classifications.items():
            if classification == category:
                priority = requirements[skill]["priority"]
                lines.append(f"- {skill} ({priority})")

    lines.extend([
        "",
        "EDUCATION SUBJECT REQUIREMENTS",
        ", ".join(education_subjects)
        if education_subjects else "None detected",
        "",
        "EXPERIENCE PATHWAYS"
    ])

    for path in pathways:
        lines.append(
            f"- {path['degree']} + {path['years']} years"
        )

    lines.extend([
        "",
        experience_result["status"],
        experience_result["message"],
        "",
        "CONTEXTUAL REQUIREMENTS"
    ])

    for item in contextual:
        lines.append(
            f"- {item['label']}: {item['status']}"
        )

    lines.extend([
        "",
        "Note: This is a local evidence-based screening aid. "
        "It does not predict hiring outcomes or determine "
        "eligibility on behalf of an employer."
    ])

    return "\n".join(lines)


# ============================================================
# 15. CANDIDATE PROFILE
# ============================================================

with st.sidebar:
    st.header("Candidate profile")

    degree = st.selectbox(
        "Highest completed qualification",
        ["Bachelor's", "Master's", "PhD", "Other"],
        index=0
    )

    degree_subject = st.text_input(
        "Degree subject",
        value="Industrial Engineering"
    )

    degree_duration = st.number_input(
        "Degree duration (years)",
        min_value=0.0,
        max_value=15.0,
        value=4.0,
        step=0.5
    )

    professional_experience = st.number_input(
        "Relevant professional experience (years)",
        min_value=0.0,
        max_value=50.0,
        value=3.5,
        step=0.5
    )

    authorisation = st.selectbox(
        "Work authorisation",
        [
            "Authorised to work in job location",
            "Require sponsorship",
            "Need to verify",
            "Not specified"
        ],
        index=0
    )

    location = st.text_input(
        "Preferred location",
        value="Sweden"
    )


# ============================================================
# 16. INPUTS
# ============================================================

st.subheader("1. Upload your CV")

uploaded_file = st.file_uploader(
    "Upload your CV (PDF)",
    type=["pdf"]
)

st.subheader("2. Add the job description")

job_text = st.text_area(
    "Complete job description",
    height=300,
    placeholder="Paste the complete job description here..."
)

analyse = st.button(
    "Analyse my CV",
    type="primary",
    use_container_width=True
)


# ============================================================
# 17. RUN ANALYSIS
# ============================================================

if analyse:

    if uploaded_file is None:
        st.error("Please upload your CV PDF.")

    elif len(job_text.strip()) < 50:
        st.error("Please paste the complete job description.")

    else:
        try:
            with st.spinner("Analysing your CV and job description..."):

                cv_text = extract_pdf(uploaded_file)
                job_text = normalize(job_text)

                if len(cv_text) < 100:
                    st.error(
                        "Not enough text could be extracted from "
                        "the PDF. It may be scanned or image-based."
                    )
                    st.stop()

                profile = {
                    "degree": degree,
                    "subject": degree_subject,
                    "duration": degree_duration,
                    "experience": professional_experience,
                    "authorisation": authorisation,
                    "location": location
                }

                cv_skills = detect_cv_skills(cv_text)

                requirements = extract_requirements(job_text)

                groups = extract_domain_groups(job_text)
                grouped = grouped_domains(groups)

                classifications = {}

                for skill, requirement in requirements.items():
                    classifications[skill] = classify_requirement(
                        skill,
                        requirement,
                        cv_skills,
                        cv_text
                    )

                overall = calculate_requirement_score(
                    requirements,
                    classifications
                )

                technical_requirements = {
                    skill: item
                    for skill, item in requirements.items()
                    if item["category"] != DOMAIN_CATEGORY
                }

                technical_classifications = {
                    skill: classifications[skill]
                    for skill in technical_requirements
                }

                technical = calculate_requirement_score(
                    technical_requirements,
                    technical_classifications
                )

                domain_requirements = {
                    skill: item
                    for skill, item in requirements.items()
                    if item["category"] == DOMAIN_CATEGORY
                }

                domain_classifications = {
                    skill: classifications[skill]
                    for skill in domain_requirements
                }

                domain = calculate_requirement_score(
                    domain_requirements,
                    domain_classifications
                )

                similarity = text_similarity(
                    cv_text,
                    job_text
                )

                pathways = extract_experience_pathways(
                    job_text
                )

                experience_result = assess_experience(
                    pathways,
                    degree,
                    professional_experience
                )

                education_subjects = extract_education_subjects(
                    job_text
                )

                contextual_raw = extract_contextual_requirements(
                    job_text
                )

                contextual = []

                for item in contextual_raw:
                    contextual.append({
                        **item,
                        "status": assess_contextual_requirement(
                            item["label"],
                            cv_text
                        )
                    })

            # ------------------------------------------------
            # PROFILE
            # ------------------------------------------------

            st.divider()
            st.header("3. Candidate profile")

            p1, p2, p3 = st.columns(3)

            p1.metric("Degree duration", f"{degree_duration} years")
            p2.metric(
                "Professional experience",
                f"{professional_experience} years"
            )
            p3.metric("Qualification", degree)

            st.write(f"**Degree subject:** {degree_subject}")

            # ------------------------------------------------
            # SCORES
            # ------------------------------------------------

            st.header("4. Match overview")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Evidence score",
                f"{overall}%"
                if overall is not None else "N/A"
            )

            c2.metric(
                "Technical score",
                f"{technical}%"
                if technical is not None else "N/A"
            )

            c3.metric(
                "Domain score",
                f"{domain}%"
                if domain is not None else "N/A"
            )

            c4.metric("Text similarity", f"{similarity}%")

            st.caption(
                "The evidence score gives full credit to direct "
                "matches, partial credit to partial matches and "
                "limited credit to transferable skills. Example "
                "technologies are excluded from the score. "
                "This is not a probability of getting hired."
            )

            # ------------------------------------------------
            # CLASSIFICATION SUMMARY
            # ------------------------------------------------

            st.header("5. Requirement classification")

            order = [
                "Direct match",
                "Partial match",
                "Transferable",
                "Domain gap",
                "Not evidenced",
                "Example technology"
            ]

            for classification in order:
                skills = [
                    skill for skill, value
                    in classifications.items()
                    if value == classification
                ]

                with st.expander(
                    f"{classification} ({len(skills)})",
                    expanded=classification in [
                        "Direct match", "Domain gap"
                    ]
                ):

                    if not skills:
                        st.write("No requirements in this category.")

                    for skill in skills:
                        requirement = requirements[skill]

                        st.markdown(
                            f"**{skill}** - {requirement['priority']}"
                        )

                        for excerpt in requirement["evidence"][:2]:
                            st.caption(excerpt[:400])

                        if classification in [
                            "Direct match", "Partial match"
                        ]:
                            evidence = get_evidence(cv_text, skill)

                            for item in evidence[:2]:
                                st.write(
                                    f"**{item['type']}:** "
                                    f"{item['text']}"
                                )

                        elif classification == "Transferable":
                            related = [
                                x for x in TRANSFERABLE.get(skill, [])
                                if x in cv_skills
                            ]

                            st.write(
                                "Related CV capabilities: "
                                + ", ".join(related)
                            )

            # ------------------------------------------------
            # DOMAIN ALTERNATIVES
            # ------------------------------------------------

            st.header("6. Alternative domain requirements")

            if groups:
                for group in groups:
                    st.write(
                        " OR ".join(group["alternatives"])
                    )

                    matched_domains = [
                        x for x in group["alternatives"]
                        if x in cv_skills
                    ]

                    if matched_domains:
                        st.success(
                            "Domain terms detected in CV: "
                            + ", ".join(matched_domains)
                        )
                    else:
                        st.warning(
                            "No direct evidence of these domains "
                            "was detected in the CV."
                        )

                    st.caption(group["evidence"])
            else:
                st.write("No OR-connected domain groups detected.")

            # ------------------------------------------------
            # CONTEXTUAL REQUIREMENTS
            # ------------------------------------------------

            st.header("7. Contextual requirements")

            if contextual:
                for item in contextual:
                    st.markdown(f"**{item['label']}**")
                    st.write(item["status"])
                    st.caption(item["evidence"])
            else:
                st.write("No additional contextual requirements detected.")

            # ------------------------------------------------
            # EDUCATION
            # ------------------------------------------------

            st.header("8. Education assessment")

            st.write(
                f"**Candidate:** {degree} in {degree_subject}"
            )

            st.write(
                f"**Degree duration:** {degree_duration} years"
            )

            if education_subjects:
                st.write(
                    "**Job's listed degree subjects:** "
                    + ", ".join(education_subjects)
                )

                st.warning(
                    "The tool cannot determine whether your degree "
                    "subject is considered equivalent to the listed "
                    "subjects. Employer review is required."
                )
            else:
                st.info(
                    "No specific degree subjects were confidently detected."
                )

            # ------------------------------------------------
            # EXPERIENCE
            # ------------------------------------------------

            st.header("9. Professional experience assessment")

            if pathways:
                for path in pathways:
                    st.write(
                        f"**{path['degree']} + {path['years']} years**"
                    )
                    st.caption(path["source"])

                st.divider()

                st.write(
                    f"**Your professional experience:** "
                    f"{professional_experience} years"
                )

                st.write(
                    f"**Assessment:** {experience_result['status']}"
                )

                st.write(experience_result["message"])

                st.caption(
                    "Degree duration is not added to professional "
                    "experience. Only relevant work experience is "
                    "compared with the job's experience threshold."
                )
            else:
                st.info(
                    "No clear degree-and-experience pathway was detected."
                )

            # ------------------------------------------------
            # WORK AUTHORISATION
            # ------------------------------------------------

            st.header("10. Work authorisation and location")

            st.write(f"**Work authorisation:** {authorisation}")
            st.write(f"**Preferred location:** {location}")

            # ------------------------------------------------
            # APPLICATION REVIEW
            # ------------------------------------------------

            st.header("11. Application review")

            if experience_result["status"] == (
                "Below detected experience threshold"
            ):
                st.warning(
                    "There may be a professional experience shortfall. "
                    "Consider applying if your experience is relevant "
                    "and the employer is flexible."
                )

            elif experience_result["status"] == (
                "Meets detected experience threshold"
            ):
                st.success(
                    "Your declared professional experience meets "
                    "a detected qualification pathway."
                )

            else:
                st.info(
                    "Verify the qualification and experience pathway "
                    "with the employer."
                )

            st.caption(
                "A screening result is not an automatic rejection "
                "recommendation. Employers make the final assessment."
            )

            # ------------------------------------------------
            # DOWNLOAD REPORT
            # ------------------------------------------------

            report = create_report(
                profile,
                overall,
                technical,
                domain,
                similarity,
                classifications,
                requirements,
                pathways,
                experience_result,
                education_subjects,
                contextual
            )

            st.download_button(
                "Download complete analysis report",
                data=report,
                file_name="jobmatch_v3_1_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as error:
            st.error("An error occurred during analysis.")
            st.exception(error)


st.divider()

st.caption(
    "JobMatch Lite V3.1 | Local analysis | "
    "No paid APIs | No external LLM"
)
