from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
from Data_Preprocessing import preprocess_text
with open(r'Resume_parse.json', 'r') as file:
    resume = json.load(file)
def calculate_role_match(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    
    # Compute Cosine Similarity
    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100
    return similarity_score

# Example Data
resume_industry_role = resume['Summary']


def get_role_match_score(data):
    hash={'Job_ID':[],'Job_Description':[],'role_match_score':[]}
    job_description = list(
    data['Role'] + ' with seniority level of ' + data['Seniority'] +
    ' with job function of ' + data['Job function'] + ' and ' + data['Job_Des'])
    for i in range(0,len(data)):
        role_match_score = calculate_role_match(resume_industry_role, job_description[i])
        print(f"Industry & Role Match Score: {role_match_score:.2f}%")
        hash['Job_ID'].append(data.iloc[i]['Job_ID'])
        hash['Job_Description'].append(data.iloc[i]["Job_Des"])
        hash['role_match_score'].append(role_match_score)
    return pd.DataFrame(hash)
