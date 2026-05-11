ME
🔍 Resume Matching Engine
Redrob AI Campus Hackathon — Individual Competition
📌 Problem Statement
Given 10 resume datasets from Indian university students and 3 Job Descriptions (JDs) from Korean technology companies, build a program that:

Normalizes noisy resume skill data
Computes TF-IDF vectors for resumes
Builds binary vectors for job descriptions
Calculates cosine similarity between resumes and JDs
Outputs the Top 3 matching candidates per JD
🗂️ Repository Structure
resume-matching-engine/
│
├── resume_matching_engine.py   # Complete solution (single file)
└── README.md                   # This file
⚙️ How It Works — Step by Step
Step 1 — Skill Normalization
Each resume's raw skill string is:

Split on commas
Lowercased
Mapped through SKILL_ALIASES (e.g. "Pyhton" → "python", "deep-learning" → "deep_learning")
Unknown tokens are silently discarded
Step 2 — Deduplication
After alias mapping, duplicate canonical skills are removed so each skill appears at most once per resume. (e.g. data-viz and matplotlib both map to data_visualization → kept once)

Step 3 — Vocabulary Construction
A shared vocabulary is built from all normalized resume skills, sorted alphabetically. This same ordering is used for both resume TF-IDF vectors and JD binary vectors.

Vocabulary size: 48 unique canonical skills

Step 4 — TF-IDF Vector Construction
For each resume, a TF-IDF vector of length 48 is computed using:

TF(skill, resume)  =  1 / N
                       where N = total unique skills in the resume

IDF(skill)         =  ln( 10 / df(skill) )
                       where df = number of resumes containing the skill
                       Natural log, no smoothing

TF-IDF             =  TF × IDF
No external libraries used — only Python's built-in math module.

Step 5 — JD Binary Vectors
JD skills (required + preferred) are normalized through the same alias map. A binary vector is built: 1.0 if the skill exists in the vocabulary, 0.0 otherwise.

Step 6 — Cosine Similarity & Ranking
Cosine(A, B) = (A · B) / (|A| × |B|)

  A = Resume TF-IDF vector
  B = JD binary vector
  |A| = Euclidean norm of A
Candidates are ranked by score (descending). Ties are broken alphabetically by name. Top 3 are selected per JD.

📊 Dataset
Resumes — 10 Candidates
ID	Candidate	Background
01	Arjun Sharma	TCS Intern · BITS Pilani CSE 2024
02	Priya Nair	Freelance Web Developer · VIT IT 2024
03	Rahul Gupta	Infosys SDE Intern · IIT Delhi 2023
04	Sneha Patel	IISc Research Assistant · IIIT Hyderabad AI 2024
05	Vikram Singh	Google SWE Intern · IIT Bombay 2024
06	Ananya Krishnan	Full Stack Developer · NIT Trichy 2022
07	Karan Mehta	Data Analyst · Delhi University 2023
08	Deepika Rao	Samsung Android Intern · NSIT 2024
09	Aditya Kumar	Frontend SDE · Flipkart / IIIT Bangalore
10	Meera Iyer	Data Science Intern · Wipro 2024
Job Descriptions — 3 Roles
JD	Company	Role
JD-1	Kakao (Seoul)	ML Engineer
JD-2	Naver (Seongnam)	Backend Engineer
JD-3	Line (Seoul)	Frontend Engineer
🚀 How to Run
Prerequisites
Python 3.x (no external libraries needed)
Run the program
python resume_matching_engine.py
or

python3 resume_matching_engine.py
What you'll see
The program prints all intermediate steps for verification, followed by the final results:

── Normalized Resume Skills ──
── Vocabulary (48 terms) ──
── IDF Values ──
── TF-IDF Vectors (non-zero only) ──
── JD Normalized Skills ──
── All Cosine Scores ──
═══════════════════════════════════════
   RESUME MATCHING ENGINE — RESULTS
═══════════════════════════════════════
✅ Final Output
JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
🔑 Key Design Decisions
Decision	Reason
Multi-word phrase matching before single tokens	Prevents "spring boot" being split into individual unknown tokens
TF = 1/N after deduplication	Each skill appears exactly once, so count is always 1
IDF = ln(10/df), no smoothing	Exactly as specified in the problem sheet
JD vectors are binary (not TF-IDF)	JDs represent requirements, not frequency-weighted documents
Vocabulary built from resumes only	JD skills not in any resume would have undefined IDF
Alphabetical tie-breaking	As required by the output rules
📚 Libraries Used
Library	Purpose
math	math.log() for IDF, math.sqrt() for vector norms
No external libraries (numpy, pandas, scikit-learn, etc.) were used, as per competition rules.

