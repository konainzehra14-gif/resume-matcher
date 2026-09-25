import os, re, datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = {
    "Python": ["python"], "JavaScript": ["javascript", "js"], "TypeScript": ["typescript"],
    "Java": ["java"], "C++": ["c++"], "C#": ["c#"], "Go": ["golang"], "PHP": ["php"],
    "SQL": ["sql"], "MongoDB": ["mongodb", "mongo"], "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"], "Redis": ["redis"], "NoSQL": ["nosql"],
    "React": ["react", "react.js", "reactjs"], "Angular": ["angular"], "Vue": ["vue", "vue.js"],
    "Node.js": ["node.js", "nodejs", "node"], "Django": ["django"], "Flask": ["flask"],
    "FastAPI": ["fastapi"], "Spring Boot": ["spring boot", "spring"], "HTML": ["html", "html5"],
    "CSS": ["css", "css3"], "REST APIs": ["rest", "restful", "rest api", "rest apis"],
    "GraphQL": ["graphql"], "Docker": ["docker"], "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"], "Azure": ["azure"], "GCP": ["gcp", "google cloud"],
    "CI/CD": ["ci/cd", "continuous integration", "jenkins", "github actions"],
    "Git": ["git", "github", "gitlab"], "Linux": ["linux"], "Terraform": ["terraform"],
    "Machine Learning": ["machine learning", "ml"], "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"], "Computer Vision": ["computer vision"],
    "TensorFlow": ["tensorflow"], "PyTorch": ["pytorch"], "scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"], "NumPy": ["numpy"], "Data Analysis": ["data analysis", "data analytics"],
    "Data Visualization": ["data visualization", "data visualisation"],
    "Tableau": ["tableau"], "Power BI": ["power bi", "powerbi"], "Spark": ["spark", "pyspark"],
    "Kafka": ["kafka"], "Airflow": ["airflow"], "Excel": ["excel"],
    "Agile": ["agile", "scrum"], "Project Management": ["project management"],
    "Leadership": ["leadership", "team lead", "mentoring"], "Communication": ["communication"],
    "Problem Solving": ["problem solving", "problem-solving"], "Testing": ["unit testing", "pytest", "jest", "tdd"],
}
GROUPS = [
    {"AWS", "Azure", "GCP"}, {"React", "Angular", "Vue"}, {"Django", "Flask", "FastAPI"},
    {"SQL", "PostgreSQL", "MySQL", "MongoDB", "NoSQL", "Redis"}, {"Docker", "Kubernetes", "Terraform"},
    {"TensorFlow", "PyTorch", "scikit-learn", "Machine Learning", "Deep Learning"},
    {"Tableau", "Power BI", "Data Visualization", "Excel"}, {"Kafka", "Spark", "Airflow"},
    {"Java", "Spring Boot"}, {"JavaScript", "TypeScript", "Node.js"}, {"Python", "Pandas", "NumPy"},
]
PATTERNS = {k: re.compile(r"(?<![\w+#.])(" + "|".join(re.escape(a) for a in v) + r")(?![\w+#])", re.I) for k, v in SKILLS.items()}
NICE = re.compile(r"nice to have|preferred|bonus|desirable|good to have|a plus", re.I)
HEAD = re.compile(r"^(requirements|qualifications|responsibilities|what you|must|about you|required)", re.I)
EDU = [("PhD", 4, r"ph\.?d|doctorate"), ("Master's", 3, r"master|m\.?sc\b|mba"),
       ("Bachelor's", 2, r"bachelor|b\.?sc\b|b\.?tech|\bbs\b|undergraduate"), ("Diploma", 1, r"diploma|associate degree")]
YEARS = re.compile(r"(\d{1,2})\+?\s*(?:years|yrs)", re.I)


def find_skills(text):
    return {k for k, p in PATTERNS.items() if p.search(text)}


def jd_skills(jd):
    must, nice, flag = set(), set(), False
    for line in jd.splitlines():
        s = line.strip()
        if not s:
            continue
        if NICE.search(s) and len(s) < 80:
            flag = True
        elif HEAD.match(s) and len(s) < 60:
            flag = False
        (nice if flag or NICE.search(s) else must).update(find_skills(s))
    return sorted(must), sorted(nice - must)


def resume_years(t):
    explicit = [int(x) for x in YEARS.findall(t)]
    now, spans = datetime.date.today().year, 0
    for a, b in re.findall(r"((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)", t, re.I):
        spans += max(0, (now if b.isalpha() else int(b)) - int(a))
    return max(max(explicit, default=0), min(spans, 40))


def edu_level(t):
    for name, lvl, p in sorted(EDU, key=lambda e: -e[1]):
        if re.search(p, t, re.I):
            return name, lvl
    return "Not found", 0


def classify(reqs, have):
    matched, partial, missing = [], [], []
    for r in reqs:
        if r in have:
            matched.append(r)
            continue
        near = next((sorted(have & g)[0] for g in GROUPS if r in g and have & g), None)
        (partial.append({"required": r, "found": near}) if near else missing.append(r))
    return matched, partial, missing


# OpenAI-style providers, tried in this order (free ones first): (key env, url, model env, default model, name)
PROVIDERS = [
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions", "GROQ_MODEL", "llama-3.3-70b-versatile", "Groq"),
    ("XAI_API_KEY", "https://api.x.ai/v1/chat/completions", "XAI_MODEL", "grok-4.3", "Grok"),
]


def llm_tips(resume, jd, matched, missing):
    """Optional AI suggestions from Groq or Grok, whichever key is set. Returns (tips, provider, error)."""
    active = next((p for p in PROVIDERS if os.getenv(p[0])), None)
    if not active:
        return [], None, None
    key_env, url, model_env, model, name = active
    safe = re.sub(r"\S+@\S+|\+?\d[\d\s().-]{8,}\d", "[redacted]", resume)[:4000]
    prompt = ("You are a resume coach. Using ONLY facts present in the resume, write 4 short, specific suggestions "
              "(bullet rewrites or keywords) to fit the job. Never invent experience. One per line, no numbering.\n"
              f"Matched: {matched}\nMissing: {missing}\nRESUME:\n{safe}\nJOB:\n{jd[:3000]}")
    try:
        import requests
        r = requests.post(url, headers={"Authorization": "Bearer " + os.getenv(key_env).strip()},
                          json={"model": os.getenv(model_env, model).strip(),
                                "messages": [{"role": "user", "content": prompt}]}, timeout=60)
        if r.status_code != 200:
            return [], None, f"{name} returned {r.status_code}: {r.text[:160]}"
        text = r.json()["choices"][0]["message"]["content"] or ""
        tips = [l.lstrip("-•* ").strip() for l in text.splitlines() if l.strip()][:4]
        return (tips, name, None) if tips else ([], None, f"{name} returned an empty answer")
    except Exception as e:
        return [], None, f"{name} request failed ({e.__class__.__name__})"


def analyze(resume, jd, tips=True):
    must, nice = jd_skills(jd)
    have = find_skills(resume)
    mm, mp, mx = classify(must, have)
    nm, npart, nx = classify(nice, have)

    total = 2 * len(must) + len(nice)
    got = 2 * (len(mm) + .5 * len(mp)) + (len(nm) + .5 * len(npart))
    skills = got / total if total else 0.5

    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([resume, jd])
    semantic = min(1.0, float(cosine_similarity(tf[0], tf[1])[0][0]) * 2)

    have_y, need_y = resume_years(resume), (YEARS.findall(jd) or [None])[0]
    need_y = int(need_y) if need_y else None
    exp = 1.0 if not need_y else min(1.0, have_y / need_y)

    have_e, need_e = edu_level(resume), edu_level(jd)
    edu = 1.0 if not need_e[1] else min(1.0, have_e[1] / need_e[1])

    score = round(100 * (.5 * skills + .2 * semantic + .2 * exp + .1 * edu))

    recs, ai_name, ai_err = llm_tips(resume, jd, mm + nm, mx) if tips else ([], None, None)
    for s in mx[:4]:
        recs.append(f"{s} is a must-have and isn't on your resume. If you've used it, add it to a role or project bullet; if not, a small project using {s} is the fastest way to close the gap.")
    for p in mp[:3]:
        recs.append(f"The role asks for {p['required']} and you show {p['found']}. Name the shared concepts so it reads as transferable experience.")
    if need_y and have_y < need_y:
        recs.append(f"The job asks for about {need_y} years; your resume shows about {have_y}. Make role dates explicit and lead with your most relevant work.")
    if need_e[1] > have_e[1]:
        recs.append(f"The job expects a {need_e[0]} degree; add your education section clearly, or highlight equivalent experience.")
    if not re.search(r"\d+\s*%|\$\s?\d|\d+x\b", resume):
        recs.append("Add measurable results (percentages, time saved, users, revenue) to your bullets. None were detected.")
    if not recs:
        recs.append("Strong fit. Mirror the job's wording for your top skills and keep bullets outcome-focused.")

    first = next((l.strip() for l in jd.splitlines() if l.strip()), "Untitled role")
    first = re.sub(r"(?i)^(job\s*title|position|role)\s*[:\-–]\s*", "", first)
    first = re.split(r"(?i)location|department|company|\||\s{2,}", first)[0].strip() or "Untitled role"
    return {
        "job_title": first[:60], "score": score,
        "breakdown": {"skills": round(skills * 100), "semantic": round(semantic * 100),
                      "experience": round(exp * 100), "education": round(edu * 100)},
        "matched": mm + nm, "partial": mp + npart, "missing": mx, "nice_missing": nx,
        "must_have": must, "extra": sorted(have - set(must) - set(nice))[:10],
        "experience": {"have": have_y, "need": need_y},
        "education": {"have": have_e[0], "need": need_e[0]},
        "recommendations": recs[:8],
        "ai_tips": ai_name, "ai_error": ai_err,
    }
