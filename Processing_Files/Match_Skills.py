import json
import pandas as pd
with open(r'Resume_parse.json', 'r') as file:
    resume = json.load(file)
def tolower(l):
    return [item.lower() for item in l]
def compute_skill_match(resume_skills, job_skills):
    matched_skills = [skill for skill in resume_skills if skill in job_skills]
    print(matched_skills)
    return len(matched_skills) / len(resume_skills)
def get_skill_score(data):
    hash={'Job_ID':[],'Job_Description':[],'skill_match_score':[]}
    for i in range(len(data)):
        job_skills = data.iloc[i]["Skills"]
        resume_skills = tolower(resume["Skills"])
        skill_match_score = compute_skill_match(resume_skills, job_skills)
        hash['Job_ID'].append(data.iloc[i]['Job_ID'])
        hash['Job_Description'].append(data.iloc[i]["Job_Des"])
        hash['skill_match_score'].append(skill_match_score)
    return pd.DataFrame(hash)
