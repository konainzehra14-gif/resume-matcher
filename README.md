# Resume Matcher

Compares a resume with a job description and returns a match score, matched skills, gaps and recommendations. Results are saved in MongoDB.

## Run it

```bash
pip install fastapi uvicorn[standard] python-multipart pymongo python-dotenv scikit-learn pdfplumber python-docx requests
# MongoDB: local install, or `docker run -d -p 27017:27017 mongo`, or an Atlas connection string
uvicorn main:app --reload
```

Open http://localhost:8000

## Optional `.env`

```
MONGO_URI=mongodb://localhost:27017      # or your Atlas URI
MONGO_DB=resume_matcher
GROQ_API_KEY=...                         # free, no card: console.groq.com (tailored suggestions, optional)
GROQ_MODEL=llama-3.3-70b-versatile
XAI_API_KEY=...                          # or Grok (console.x.ai, paid after promo credits)
XAI_MODEL=grok-4.3                       # change if xAI renames models
JSEARCH_API_KEY=                         # optional. Without it, free Remotive remote jobs are used
```

## How scoring works

`score = 0.50 skills + 0.20 wording similarity (TF-IDF cosine) + 0.20 experience + 0.10 education`

Must-have skills count double; skills in the same family (e.g. AWS vs GCP) count as half credit and appear as "Related only".

## Files

- `matcher.py`: skill taxonomy, extraction, scoring, gap analysis, recommendations
- `main.py`: FastAPI routes and MongoDB storage (`analyses` collection)
- `static/index.html`: the UI

## Privacy

Only the analysis result is stored in MongoDB, never the resume text. When an API key is set, resume text is sent to the Claude API with emails and phone numbers removed.

## Next steps

Grow the `SKILLS` list (or load ESCO), swap TF-IDF for `sentence-transformers`, and hand-label 30 to 50 pairs to check your score against.
