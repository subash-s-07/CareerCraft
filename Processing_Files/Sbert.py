from sentence_transformers import SentenceTransformer
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from Data_Preprocessing import preprocess_text
model = SentenceTransformer('all-MiniLM-L6-v2')
with open(r'Resume_parse.json', 'r') as file:
    resume = json.load(file)
# Convert Resume & Job Description to Embeddings
def similarity_score_process(data):
    resume_text = preprocess_text(resume['summary_paragraph'])
    resume_embedding = model.encode(resume_text)
    hash={'Job_ID':[],'Job_Description':[],'similarity_score':[]}
    for i in range(len(data)):
        job_description = preprocess_text(data.iloc[i]["Job_Des"])
        job_embedding = model.encode(job_description)
        
        similarity_score = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        hash['Job_ID'].append(data.iloc[i]['Job_ID'])
        hash['Job_Description'].append(data.iloc[i]["Job_Des"])
        hash['similarity_score'].append(similarity_score)
    return pd.DataFrame(hash)

