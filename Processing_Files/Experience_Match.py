import re
import json
import pandas as pd
with open(r'Resume_parse.json', 'r') as file:
    resume = json.load(file)
def extract_experience_from_text(text):
    # Extract experience requirement using regex (e.g., "4+ years", "2 years")
    match = re.search(r"(\d+)\+?\s*years?", text)
    return int(match.group(1)) if match else 0

def calculate_experience_match(resume_experience, job_description):
    job_experience_required = extract_experience_from_text(job_description)
    
    # Score based on how well resume experience meets the requirement
    if job_experience_required == 0:
        return 100  # No specific experience required

    experience_match_score = (resume_experience / job_experience_required) * 100
    return min(experience_match_score, 100)  # Cap at 100%

# Example Data
resume_experience = resume["Total Work Experience in Years"]

def get_exp_score(data):
    hash={'Job_ID':[],'Job_Description':[],'exp_score':[]}
    job_description = list(data['Job_Des'])
    for i in range(len(data)):
        exp_score = calculate_experience_match(resume_experience,job_description [i])
        print(f"Experience Match Score: {exp_score:.2f}%")
        hash['Job_ID'].append(data.iloc[i]['Job_ID'])
        hash['Job_Description'].append(data.iloc[i]["Job_Des"])
        hash['exp_score'].append(exp_score)
    return pd.DataFrame(hash)
        
