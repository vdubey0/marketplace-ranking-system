"""Generate structured relevance truth and performance outcomes without API calls."""
from __future__ import annotations

import argparse, hashlib, json, logging, math, re, sys, unicodedata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
ACQ = ROOT / "data-acquisition"
sys.path.insert(0, str(ACQ))
from scripts.authentic.load_data import authentic_data_directory, data_directory, file_hash
from scripts.synthetic.io_utils import read_jsonl
from scripts.synthetic.models import CandidateSpec

DEFAULT_CONFIG = ROOT / "ground-truth-generation/config/v1.json"
DEFAULT_OUTPUT = ROOT / "ground-truth-generation/data/v1"
LEVEL = {"entry": 0, "mid": 1, "senior": 2, "leadership": 3}
SOURCE_LEVEL = {"Internship": "entry", "Entry level": "entry", "Associate": "mid",
                "Mid-Senior level": "senior", "Director": "leadership", "Executive": "leadership"}
FIRST = ["Alex","Avery","Blake","Cameron","Casey","Charlie","Dakota","Drew","Elliot","Emerson",
         "Finley","Harper","Hayden","Jamie","Jordan","Kai","Kendall","Lane","Logan","Morgan",
         "Parker","Quinn","Reese","Remy","Riley","River","Robin","Rowan","Sage","Sam","Taylor","Toni",
         "Adrian","Ari","Bailey","Corey","Devon","Ellis","Jesse","Marley","Micah","Noel","Payton","Shawn",
         "Skyler","Sydney","Terry","Tracy","Winter","Wren"]
LAST = ["Adams","Allen","Baker","Bell","Bennett","Brooks","Campbell","Carter","Clark","Collins",
        "Cooper","Davis","Edwards","Evans","Foster","Garcia","Gray","Green","Hall","Harris","Hayes",
        "Hill","Howard","Jackson","James","Kelly","Lee","Lewis","Martin","Miller","Moore","Morgan",
        "Nelson","Parker","Perez","Price","Reed","Rivera","Roberts","Ross","Scott","Smith","Stewart",
        "Thomas","Turner","Walker","Ward","White","Williams","Young"]
DOMAINS = {"healthcare":("hospital","patient","clinical","medical","healthcare"),
           "finance":("bank","fintech","financial","insurance","investment"),
           "technology":("software","technology","cloud","saas","information technology"),
           "education":("school","university","college","education","academic"),
           "manufacturing":("manufacturing","factory","plant","production"),
           "retail":("retail","store","merchandising"), "government":("government","federal","municipal"),
           "hospitality":("hotel","restaurant","hospitality","food service"),
           "construction":("construction","contractor","building"),
           "logistics":("logistics","warehouse","transportation","distribution")}

def norm(value) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)): return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9+#.]+", " ", text).strip()

def domain(text: str) -> str:
    text = norm(text); scores = {k: sum(v in text for v in values) for k, values in DOMAINS.items()}
    best = max(scores, key=scores.get); return best if scores[best] else "unknown"

def edu(value: str) -> str:
    value = norm(value)
    for key, label in [("doctor","doctorate"),("phd","doctorate"),("master","masters"),
                       ("mba","masters"),("bachelor","bachelors"),("associate","associates")]:
        if key in value: return label
    return "other_or_unspecified"

def seed_for(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "big")

def deterministic_uniform(keys: pd.Series, salt: str) -> np.ndarray:
    """Return reproducible U[0, 1) values from an independently salted stream."""
    hashes = pd.util.hash_pandas_object(keys + "|" + salt, index=False).to_numpy(dtype="uint64")
    # Keep the 53 bits that float64 can represent exactly.
    return (hashes >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

def candidates_table(out: Path) -> pd.DataFrame:
    profiles = [json.loads(p.read_text()) for p in sorted((data_directory()/"synthetic/generated/resumes/v1/profiles").glob("*.json"))]
    names = [f"{a} {b}" for a in FIRST for b in LAST]
    names += [f"{a} {letter}. {b}" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for a in FIRST for b in LAST]
    if len(names) < len(profiles): raise ValueError("Name pool is smaller than candidate pool")
    np.random.default_rng(20260910).shuffle(names)
    rows=[]
    for profile, name in zip(profiles, names):
        s=CandidateSpec.model_validate(profile["spec"]); industries=sorted({x.industry for x in s.experience})
        rows.append({"candidate_id":s.candidate_id,"name":name,"resume_text":profile["resume_text"],
          "current_title":s.target_title,"skills":s.skills,"years_experience":s.years_experience,
          "education_level":edu(s.education[0].credential),"industries":industries,"domain":domain(" ".join(industries)),
          "role_family":s.role_family,"role_type":s.role_type,"role_track":s.role_track,"seniority":s.seniority,
          "certifications":s.certifications,"is_synthetic":True,"fit_profile":s.fit_profile,
          "label_status":"labeled","source":"synthetic_resume","source_resume_id":None})
    authentic_path=data_directory()/"authentic/processed/candidate_profiles_v1/candidates.parquet"
    if authentic_path.exists():
        authentic=pd.read_parquet(authentic_path)
        authentic["domain"]=authentic.industries.map(lambda values:domain(" ".join(values)))
        rows.extend(authentic.to_dict("records"))
    frame=pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    frame.to_parquet(out/"candidates.parquet",index=False); return frame

def seniority(title, supplied) -> str:
    title=norm(title)
    if re.search(r"\b(chief|vice president|vp|director|head of)\b",title): return "leadership"
    if re.search(r"\b(senior|sr|lead|principal|manager|supervisor)\b",title): return "senior"
    if re.search(r"\b(junior|jr|assistant|intern|trainee)\b",title): return "entry"
    return SOURCE_LEVEL.get(str(supplied),"mid")

def min_years(text: str):
    values=[int(x) for x in re.findall(r"\b(\d{1,2})(?:\s*(?:-|to)\s*\d{1,2})?\s*\+?\s*years?\b",norm(text))]
    values=[x for x in values if 0<x<=25]; return float(max(values)) if values else np.nan

def jobs_table(out: Path, limit=None) -> pd.DataFrame:
    source=authentic_data_directory()/"processed"
    mapping=pd.read_parquet(source/"job_role_mapping.parquet")
    mapping=mapping.loc[mapping.assignment_status.eq("assigned"),["job_id","role_family","role_type"]]
    if limit: mapping=mapping.head(limit)
    jobs=pd.read_parquet(source/"jobs.parquet",columns=["job_id","title","description","skills_desc","formatted_experience_level"])
    frame=mapping.merge(jobs,on="job_id",validate="one_to_one")
    catalog={(x["role_family"],x["role_type"]):x for x in read_jsonl(data_directory()/"synthetic/generated/role_catalog/v1/role_archetypes.jsonl")}
    rows=[]
    for x in tqdm(frame.itertuples(index=False),total=len(frame),desc="Normalize jobs",unit="job"):
        text=norm(f"{x.title} {x.skills_desc} {x.description}"); title=norm(x.title); best=None; best_score=-1
        for track in catalog[(x.role_family,x.role_type)]["tracks"]:
            target=max((len(set(title.split())&set(norm(t).split()))/max(1,len(set(title.split())|set(norm(t).split()))) for t in track["target_titles"]),default=0)
            hit=sum(norm(s) in text for s in track["core_skills"]); score=3*target+hit/max(1,len(track["core_skills"]))
            if score>best_score: best,best_score=track,score
        found=[s for s in best["core_skills"]+best["optional_skills"] if norm(s) in text]
        preferred=[]; required=[]
        for skill in dict.fromkeys(found):
            pos=text.find(norm(skill)); context=text[max(0,pos-90):pos+len(norm(skill))+90]
            (preferred if re.search(r"\b(preferred|nice to have|plus|desirable)\b",context) else required).append(skill)
        requirement=x.skills_desc if not pd.isna(x.skills_desc) else str(x.description)[:5000]
        certs=[s for s in best["certifications"] if norm(s) in text]
        rows.append({"job_id":str(x.job_id),"title":str(x.title),"description":str(x.description),
          "role_family":x.role_family,"role_type":x.role_type,"role_track":best["track_name"],
          "track_assignment_score":round(best_score,4),"seniority":seniority(x.title,x.formatted_experience_level),
          "minimum_years_experience":min_years(requirement),"required_skills":required,
          "preferred_skills":preferred,"required_certifications":certs,"domain":domain(text[:8000])})
    result=pd.DataFrame(rows).sort_values("job_id").reset_index(drop=True)
    result.to_parquet(out/"jobs.parquet",index=False); return result

class Writers:
    def __init__(self,out):
        self.paths={"positive":out/"relevance_ground_truth.parquet","negative":out/"relevance_negative_sample.parquet"}
        self.w={};self.n={k:0 for k in self.paths};self.buffers={k:[] for k in self.paths};self.buffered={k:0 for k in self.paths}
    def add(self,key,frame):
        if frame.empty:return
        self.buffers[key].append(frame);self.buffered[key]+=len(frame)
        if self.buffered[key]<100000:return
        self.flush(key)
    def flush(self,key):
        if not self.buffers[key]:return
        frame=pd.concat(self.buffers[key],ignore_index=True);self.buffers[key]=[];self.buffered[key]=0
        table=pa.Table.from_pandas(frame,preserve_index=False)
        if key not in self.w:self.w[key]=pq.ParquetWriter(self.paths[key],table.schema,compression="zstd")
        self.w[key].write_table(table);self.n[key]+=len(frame)
    def close(self):
        for key in self.paths:self.flush(key)
        for w in self.w.values():w.close()

def relevance(cands,jobs,out,cfg):
    writers=Writers(out); audit={0:pd.DataFrame(),1:pd.DataFrame(),2:pd.DataFrame()}; ids=cands.candidate_id.to_numpy()
    families=cands.role_family.to_numpy(); roles=cands.role_type.to_numpy(); tracks=cands.role_track.to_numpy()
    levels=cands.seniority.map(LEVEL).to_numpy(); years=cands.years_experience.to_numpy(float); domains=cands.domain.to_numpy()
    skills=[{norm(s) for s in x} for x in cands.skills]; certs=[{norm(s) for s in x} for x in cands.certifications]
    w=cfg["weights"]; clear=cfg["thresholds"]["clearly_relevant"]; border=cfg["thresholds"]["borderline"]
    try:
      for job in tqdm(jobs.itertuples(index=False),total=len(jobs),desc="Score relevance",unit="job"):
        pool=np.flatnonzero(families==job.role_family); req={norm(x) for x in job.required_skills}; pref={norm(x) for x in job.preferred_skills}
        skill=np.array([len(skills[i]&req)/len(req) if req else .6 for i in pool])
        if pref:skill=.8*skill+.2*np.array([len(skills[i]&pref)/len(pref) for i in pool])
        occ=np.where((roles[pool]==job.role_type)&(tracks[pool]==job.role_track),1,np.where(roles[pool]==job.role_type,.75,.45))
        gap=levels[pool]-LEVEL[job.seniority]; sen=np.select([gap==0,gap==1,gap>1,gap==-1],[1,.9,.75,.5],default=.15)
        exp=np.full(len(pool),.75) if pd.isna(job.minimum_years_experience) else np.minimum(1,years[pool]/max(1,job.minimum_years_experience))
        dom=np.where(job.domain=="unknown",.5,np.where(domains[pool]==job.domain,1,np.where(domains[pool]=="unknown",.5,.2)))
        score=w["skills"]*skill+w["occupation"]*occ+w["seniority"]*sen+w["experience"]*exp+w["domain"]*dom
        needed={norm(x) for x in job.required_certifications}; hard=np.array([bool(needed and not needed.issubset(certs[i])) for i in pool])
        grade=np.where(~hard&(score>=clear),2,np.where(~hard&(score>=border),1,0)).astype("int8")
        base=pd.DataFrame({"job_id":job.job_id,"candidate_id":ids[pool],"relevance_grade":grade,"relevant":grade>0,
          "relevance_score":score.round(4),"skill_score":skill.round(4),"occupation_score":occ,
          "seniority_score":sen,"experience_score":exp.round(4),"domain_score":dom,"hard_gate_failed":hard})
        writers.add("positive",base.loc[grade>0])
        outside=np.flatnonzero(families!=job.role_family); rng=np.random.default_rng(seed_for(job.job_id,cfg["seed"])); neg=[]
        inside=np.flatnonzero(grade==0); hardest=inside[np.argsort(score[inside])[-min(2,len(inside)):]] if len(inside) else []
        if len(hardest):neg.append(base.iloc[hardest])
        take=cfg["negative_pairs_per_job"]-sum(len(x) for x in neg)
        pick=rng.choice(outside,min(take,len(outside)),replace=False)
        if len(pick):neg.append(pd.DataFrame({"job_id":job.job_id,"candidate_id":ids[pick],"relevance_grade":np.int8(0),"relevant":False,
          "relevance_score":0.0,"skill_score":0.0,"occupation_score":0.0,"seniority_score":0.0,
          "experience_score":0.0,"domain_score":0.0,"hard_gate_failed":True}))
        negative=pd.concat(neg,ignore_index=True);writers.add("negative",negative)
        for grade_value,part in [(0,negative),(1,base.loc[grade==1]),(2,base.loc[grade==2])]:
          if len(part):
            p=part.copy();p["_priority"]=pd.util.hash_pandas_object(p.job_id+"|"+p.candidate_id+f"|{cfg['seed']}",index=False).to_numpy()
            audit[grade_value]=pd.concat([audit[grade_value],p]).nsmallest(cfg["audit_pairs_per_grade"],"_priority")
    finally:writers.close()
    review=pd.concat(audit.values(),ignore_index=True).drop(columns="_priority")
    review=review.merge(jobs[["job_id","title"]],on="job_id").merge(cands[["candidate_id","name","current_title"]],on="candidate_id")
    review["reviewer_grade"]="";review["reviewer_notes"]="";review.to_csv(out/"relevance_audit_sample.csv",index=False)
    return {"relevant_or_borderline_pairs":writers.n["positive"],"negative_sample_pairs":writers.n["negative"],"audit_pairs":len(review)}

def hidden(cands,jobs,out,seed):
    rng=np.random.default_rng(seed);base=cands.seniority.map({"entry":.45,"mid":.57,"senior":.68,"leadership":.75}).to_numpy()
    ch=pd.DataFrame({"candidate_id":cands.candidate_id,"name":cands.name,"latent_ability":np.clip(base+rng.normal(0,.14,len(cands)),.05,.98),
      "latent_reliability":rng.beta(5,2.5,len(cands)),"latent_domain_depth":np.clip(base+rng.normal(0,.18,len(cands)),.03,.99),
      "latent_skill_mastery":np.clip(.55+.25*base+rng.normal(0,.14,len(cands)),.05,.99),"simulation_seed":seed})
    jbase=jobs.seniority.map({"entry":.25,"mid":.45,"senior":.65,"leadership":.8}).to_numpy();years=jobs.minimum_years_experience.fillna(3).clip(0,15).to_numpy()/30
    count=jobs.required_skills.map(len).to_numpy();jh=pd.DataFrame({"job_id":jobs.job_id,
      "latent_difficulty":np.clip(.1+.55*jbase+years+rng.normal(0,.12,len(jobs)),.03,.98),
      "latent_specialization":np.clip(.2+.08*np.minimum(count,7)+jobs.required_certifications.map(bool).to_numpy()*.15+rng.normal(0,.12,len(jobs)),.03,.98),"simulation_seed":seed})
    ch.to_parquet(out/"candidate_hidden.parquet",index=False);jh.to_parquet(out/"job_hidden.parquet",index=False);return ch,jh

def simulate_pair_outcomes(f, ch, jh, seed):
    """Evaluate supplied pair fit scores using the existing deterministic simulator."""
    c = ch.set_index('candidate_id').loc[f.candidate_id]
    j = jh.set_index('job_id').loc[f.job_id]
    fit = f.skill_score.to_numpy() * c.latent_skill_mastery.to_numpy()
    keys = f.job_id + '|' + f.candidate_id + f'|{seed}'
    noise = np.sin(deterministic_uniform(keys, 'probability_noise_v2') * 2 * math.pi) * .25
    z = (-2 + 1.1 * fit + .65 * f.occupation_score.to_numpy()
         + .35 * f.experience_score.to_numpy() + 1.15 * c.latent_ability.to_numpy()
         + .8 * c.latent_reliability.to_numpy() + .8 * c.latent_domain_depth.to_numpy()
         - 1.25 * j.latent_difficulty.to_numpy()
         - .65 * j.latent_specialization.to_numpy() * (1 - fit) + noise)
    probability = 1 / (1 + np.exp(-z))
    return pd.DataFrame({'job_id': f.job_id.to_numpy(), 'candidate_id': f.candidate_id.to_numpy(),
                         'true_success_probability': probability.round(6),
                         'successful_performance': deterministic_uniform(keys, 'realized_outcome_v2') < probability})

def outcomes(out,ch,jh,seed):
    writer=None;total=0
    final_path=out/"marketplace_outcomes.parquet";temporary_path=out/"marketplace_outcomes.tmp.parquet"
    try:
      for path in [out/"relevance_ground_truth.parquet",out/"relevance_negative_sample.parquet"]:
       for batch in tqdm(pq.ParquetFile(path).iter_batches(100000),desc=f"Simulate {path.stem}",unit="batch"):
        f=batch.to_pandas(); simulated=simulate_pair_outcomes(f,ch,jh,seed)
        prob=simulated.true_success_probability.to_numpy();outcome=simulated.successful_performance.to_numpy()
        result=pd.DataFrame({"job_id":f.job_id.to_numpy(),"candidate_id":f.candidate_id.to_numpy(),"relevance_grade":f.relevance_grade.to_numpy(),"true_success_probability":prob.round(6),"potential_success":outcome,"successful_performance":outcome,"selected":pd.array([pd.NA]*len(f),dtype="boolean"),"observed_success":pd.array([pd.NA]*len(f),dtype="boolean"),"simulation_version":"v1"})
        table=pa.Table.from_pandas(result,preserve_index=False)
        if writer is None:writer=pq.ParquetWriter(temporary_path,table.schema,compression="zstd")
        writer.write_table(table);total+=len(result)
    finally:
      if writer:writer.close()
    temporary_path.replace(final_path)
    return total

def run(config_path,out,job_limit=None):
    cfg=json.loads(config_path.read_text());out.mkdir(parents=True,exist_ok=True);c=candidates_table(out);j=jobs_table(out,job_limit)
    rel=relevance(c,j,out,cfg);ch,jh=hidden(c,j,out,cfg["seed"]);n=outcomes(out,ch,jh,cfg["seed"])
    manifest={"version":cfg["version"],"created_at_utc":datetime.now(timezone.utc).isoformat(),"candidate_count":len(c),
      "synthetic_candidate_count":int(c.is_synthetic.sum()),"authentic_candidate_count":int((~c.is_synthetic).sum()),
      "job_count":len(j),**rel,"outcome_pairs":n,"config_sha256":file_hash(config_path),"hidden_columns_are_model_features":False}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=DEFAULT_CONFIG);p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);p.add_argument("--job-limit",type=int);a=p.parse_args()
    logging.basicConfig(level=logging.INFO);print(json.dumps(run(a.config,a.output_dir,a.job_limit),indent=2))
if __name__=="__main__":main()
