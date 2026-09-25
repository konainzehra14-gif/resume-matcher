import io, os, re, time, html as ihtml, uuid, datetime as dt
from collections import OrderedDict
import requests
from pathlib import Path
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError

from matcher import analyze

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=3000)
col = client[os.getenv("MONGO_DB", "resume_matcher")]["analyses"]
app = FastAPI(title="Resume Matcher")


@app.middleware("http")
async def no_cache(request, call_next):  # always serve the latest UI file
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp

RESUMES = OrderedDict()  # in memory only, never written to MongoDB


def read_resume(file: UploadFile | None, text: str) -> str:
    if file and file.filename:
        data, name = file.file.read(), file.filename.lower()
        if name.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        if name.endswith(".docx"):
            import docx
            return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
        return data.decode("utf-8", "ignore")
    return text


def oid(id_: str) -> ObjectId:
    try:
        return ObjectId(id_)
    except InvalidId:
        raise HTTPException(404, "Analysis not found.")


@app.get("/api/health")
def health():
    try:
        client.admin.command("ping")
        return {"mongo": True}
    except PyMongoError:
        return {"mongo": False}


@app.post("/api/analyze")
def analyze_endpoint(job_description: str = Form(...), resume_text: str = Form(""),
                     resume_file: UploadFile | None = File(None)):
    resume = read_resume(resume_file, resume_text)
    if len(resume.strip()) < 50:
        raise HTTPException(400, "The resume is empty or too short. Upload a PDF/DOCX or paste at least a few lines.")
    if len(job_description.strip()) < 50:
        raise HTTPException(400, "The job description is too short. Paste the full posting.")
    result = analyze(resume, job_description)
    saved_id = None
    try:  # privacy: only the analysis is stored, never the resume text
        saved_id = str(col.insert_one({"title": result["job_title"], "created_at": dt.datetime.utcnow(),
                                       "result": result}).inserted_id)
    except PyMongoError:
        pass
    key = uuid.uuid4().hex
    RESUMES[key] = resume
    while len(RESUMES) > 20:
        RESUMES.popitem(last=False)
    return {"id": saved_id, "saved": saved_id is not None, "result": result, "resume_key": key}


CACHE = {}


def remotive(query, location):
    """Free, no key. Remote jobs from remotive.com (asks for max ~4 calls/day, so we cache)."""
    def get(q):
        hit = CACHE.get(q)
        if hit and time.time() - hit[0] < 900:
            return hit[1]
        r = requests.get("https://remotive.com/api/remote-jobs", params={"search": q, "limit": 50}, timeout=30)
        r.raise_for_status()
        jobs = r.json().get("jobs", [])
        CACHE[q] = (time.time(), jobs)
        return jobs
    jobs = get(query)
    if not jobs and " " in query.strip():  # broaden: retry with the longest word
        jobs = get(max(query.split(), key=len))
    ok = re.compile(r"worldwide|anywhere|asia|pakistan|" + re.escape(location.strip().lower() or "worldwide"), re.I)
    out = []
    for j in jobs:
        where = j.get("candidate_required_location") or ""
        if location.strip() and where and not ok.search(where):
            continue
        desc = ihtml.unescape(re.sub(r"<[^>]+>", " ", j.get("description") or ""))
        out.append({"title": j.get("title"), "company": j.get("company_name"), "location": where,
                    "source": "Remotive", "link": j.get("url"), "desc": desc})
    return out


def jsearch(api_key, query, location, linkedin_only):
    q = query.strip() + (f" in {location.strip()}" if location.strip() else "") + (" via linkedin" if linkedin_only else "")
    r = requests.get("https://api.openwebninja.com/jsearch/search", headers={"x-api-key": api_key},
                     params={"query": q, "page": 1, "num_pages": 1, "date_posted": "month"}, timeout=40)
    r.raise_for_status()
    return [{"title": j.get("job_title"), "company": j.get("employer_name"),
             "location": j.get("job_location") or j.get("job_country") or "", "source": j.get("job_publisher", ""),
             "link": j.get("job_apply_link"), "desc": j.get("job_description")} for j in r.json().get("data", [])]


@app.post("/api/jobs")
def find_jobs(resume_key: str = Form(...), query: str = Form(...), location: str = Form(""),
              linkedin_only: bool = Form(True)):
    resume = RESUMES.get(resume_key)
    if not resume:
        raise HTTPException(400, "Run a new analysis first. Resume text is kept only in memory, so it's gone after a restart.")
    api_key = os.getenv("JSEARCH_API_KEY")  # optional: without it, free Remotive jobs are used
    try:
        raw = jsearch(api_key, query, location, linkedin_only) if api_key else remotive(query, location)
    except Exception as e:
        raise HTTPException(502, f"The job search failed ({e.__class__.__name__}). Wait a minute and try again.")
    jobs = []
    for j in raw:
        link, desc = j.get("link") or "", j.get("desc") or ""
        if not link.startswith("http") or len(desc) < 50:
            continue
        a = analyze(resume, desc, tips=False)
        jobs.append({"title": j["title"] or "Untitled", "company": j["company"] or "", "location": j["location"],
                     "source": j["source"], "link": link, "score": a["score"],
                     "matched": a["matched"][:5], "missing": a["missing"][:4]})
    jobs.sort(key=lambda x: -x["score"])
    jobs = jobs[:10]
    try:
        col.database["job_searches"].insert_one({"query": query, "created_at": dt.datetime.utcnow(), "jobs": jobs})
    except PyMongoError:
        pass
    return {"provider": "JSearch" if api_key else "Remotive", "jobs": jobs}


@app.get("/api/analyses")
def list_analyses():
    try:
        rows = col.find({}, {"title": 1, "created_at": 1, "result.score": 1}).sort("created_at", DESCENDING).limit(30)
        return [{"id": str(r["_id"]), "title": r["title"], "score": r["result"]["score"],
                 "created_at": r["created_at"].isoformat() + "Z"} for r in rows]
    except PyMongoError:
        return []


@app.get("/api/analyses/{id_}")
def get_analysis(id_: str):
    doc = col.find_one({"_id": oid(id_)})
    if not doc:
        raise HTTPException(404, "Analysis not found.")
    return {"id": id_, "result": doc["result"]}


@app.delete("/api/analyses/{id_}")
def delete_analysis(id_: str):
    col.delete_one({"_id": oid(id_)})
    return {"deleted": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
