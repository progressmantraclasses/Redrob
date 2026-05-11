
import math

# ─────────────────────────────────────────────────────────────
# SKILL_ALIASES  —  use exactly as provided, do NOT modify
# ─────────────────────────────────────────────────────────────
SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",

    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",

    # Web — Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",

    # Web — Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",

    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",

    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",

    # Mobile
    "android": "android",
    "firebase": "firebase",

    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",

    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}

# ─────────────────────────────────────────────────────────────
# RAW RESUME DATASET  —  10 candidates
# ─────────────────────────────────────────────────────────────
RESUMES = {
    "Arjun Sharma":    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",
    "Priya Nair":      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",
    "Rahul Gupta":     "Java, Spring Boot, MySql, Microservices, Docker, kubernates",
    "Sneha Patel":     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",
    "Vikram Singh":    "C++, Algoritms, Data Structure, competitive programming, python",
    "Ananya Krishnan": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",
    "Karan Mehta":     "Python, Sklearn, XGboost, feature engineering, SQL, tableau",
    "Deepika Rao":     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",
    "Aditya Kumar":    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",
    "Meera Iyer":      "python, R, statistics, ML, regression, clustering, Power-BI",
}

# ─────────────────────────────────────────────────────────────
# JOB DESCRIPTIONS  —  required + preferred skills combined
# ─────────────────────────────────────────────────────────────
JDS = {
    "JD-1": {
        "label": "Kakao (ML Engineer)",
        "skills": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, "
                  "Data Visualization, NLP, BERT, Feature Engineering, Statistics",
    },
    "JD-2": {
        "label": "Naver (Backend Engineer)",
        "skills": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, "
                  "Kubernetes, REST API, CI/CD, Redis",
    },
    "JD-3": {
        "label": "Line (Frontend Engineer)",
        "skills": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, "
                  "Node.js, GraphQL, Redux, Jest, AWS",
    },
}

# ─────────────────────────────────────────────────────────────
# STEP 1 & 2 — Normalize + Deduplicate
# ─────────────────────────────────────────────────────────────
def normalize_skills(raw_string):
    """
    Split on commas → lowercase → alias lookup → deduplicate.
    Tokens absent from SKILL_ALIASES are silently discarded.
    Returns an ordered list of unique canonical skill names.
    """
    tokens = [t.strip().lower() for t in raw_string.split(",")]
    seen = set()
    canonical = []
    for token in tokens:
        mapped = SKILL_ALIASES.get(token)
        if mapped and mapped not in seen:
            seen.add(mapped)
            canonical.append(mapped)
    return canonical


# Build normalized resume skill lists
normalized_resumes = {name: normalize_skills(raw) for name, raw in RESUMES.items()}

# ─────────────────────────────────────────────────────────────
# STEP 3 — Build Shared Vocabulary
# ─────────────────────────────────────────────────────────────
all_skills = set()
for skills in normalized_resumes.values():
    all_skills.update(skills)

VOCAB = sorted(all_skills)          # alphabetically sorted
VOCAB_INDEX = {s: i for i, s in enumerate(VOCAB)}
V = len(VOCAB)                       # vocabulary size = 48

# ─────────────────────────────────────────────────────────────
# STEP 4 — Compute TF-IDF Vectors for Resumes
# Formula: TF = 1/N  |  IDF = ln(10 / df)  |  TF-IDF = TF * IDF
# ─────────────────────────────────────────────────────────────
N_DOCS = 10   # total resumes in corpus

# Document frequency: how many resumes contain each skill
df = {s: sum(1 for skills in normalized_resumes.values() if s in skills)
      for s in VOCAB}

# IDF (natural log, no smoothing)
idf = {s: math.log(N_DOCS / df[s]) for s in VOCAB}

def build_tfidf_vector(skills):
    """
    Given a deduplicated skill list (length N), return a TF-IDF vector
    of length V over the shared vocabulary.
    TF = 1/N for every present skill (post-deduplication count is always 1).
    """
    N = len(skills)
    vec = [0.0] * V
    for s in skills:
        tf = 1.0 / N
        vec[VOCAB_INDEX[s]] = tf * idf[s]
    return vec


tfidf_vectors = {name: build_tfidf_vector(skills)
                 for name, skills in normalized_resumes.items()}

# ─────────────────────────────────────────────────────────────
# STEP 5 — Build Binary JD Vectors
# ─────────────────────────────────────────────────────────────
def build_jd_vector(raw_jd_skills):
    """
    Normalize JD skills through the same alias map, then set vector
    position to 1.0 for every skill that exists in the resume vocabulary.
    """
    jd_skills = normalize_skills(raw_jd_skills)
    vec = [0.0] * V
    for s in jd_skills:
        if s in VOCAB_INDEX:
            vec[VOCAB_INDEX[s]] = 1.0
    return vec


jd_vectors = {jid: build_jd_vector(info["skills"]) for jid, info in JDS.items()}

# ─────────────────────────────────────────────────────────────
# STEP 6 — Cosine Similarity & Top-3 Ranking
# Formula: Cosine(A, B) = (A · B) / (|A| × |B|)
# ─────────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_candidates(jd_vec):
    """
    Score every resume against the JD vector.
    Sort by score descending; break ties alphabetically by name.
    """
    scores = [
        (name, cosine_similarity(resume_vec, jd_vec))
        for name, resume_vec in tfidf_vectors.items()
    ]
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


# ─────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────
def print_results():
    print("=" * 55)
    print("   RESUME MATCHING ENGINE — RESULTS")
    print("=" * 55)

    for jid, info in JDS.items():
        ranked = rank_candidates(jd_vectors[jid])
        top3   = ranked[:3]
        print(f"\n{jid} — {info['label']}")
        print(", ".join(f"{name}({score:.2f})" for name, score in top3))

    print("\n" + "=" * 55)


def print_debug():
    """Intermediate values for verification."""
    print("\n── Normalized Resume Skills ──")
    for name, skills in normalized_resumes.items():
        print(f"  {name:20s}: {skills}")

    print(f"\n── Vocabulary ({V} terms) ──")
    print(VOCAB)

    print("\n── IDF Values ──")
    for s in VOCAB:
        print(f"  {s:30s}: df={df[s]:2d}  idf={idf[s]:.6f}")

    print("\n── TF-IDF Vectors (non-zero only) ──")
    for name, vec in tfidf_vectors.items():
        entries = [(VOCAB[i], round(v, 6)) for i, v in enumerate(vec) if v > 0]
        print(f"  {name}: {entries}")

    print("\n── JD Normalized Skills ──")
    for jid, info in JDS.items():
        active = [VOCAB[i] for i, v in enumerate(jd_vectors[jid]) if v > 0]
        print(f"  {jid}: {active}")

    print("\n── All Cosine Scores ──")
    for jid, info in JDS.items():
        print(f"\n  {jid} — {info['label']}")
        ranked = rank_candidates(jd_vectors[jid])
        for name, score in ranked:
            print(f"    {name:20s}: {score:.6f}")


if __name__ == "__main__":
    print_debug()
    print_results()
