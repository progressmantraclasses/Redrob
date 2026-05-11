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
