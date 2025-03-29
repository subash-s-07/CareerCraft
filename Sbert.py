from sentence_transformers import SentenceTransformer
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from Data_Preprocessing import preprocess_text
model = SentenceTransformer('all-MiniLM-L6-v2')
with open(r'Resume_parse.json', 'r') as file:
    resume = json.load(file)
data=pd.read_csv(r'jobs.csv')
job_data=data.iloc[4]
# Convert Resume & Job Description to Embeddings
resume_text = preprocess_text(resume['summary_paragraph'])
resume_embedding = model.encode(resume_text)

job_description = preprocess_text(job_data["Job_Des"])
job_embedding = model.encode(job_description)
 
similarity_score = cosine_similarity([resume_embedding], [job_embedding])[0][0]

print(f"Job Relevance Score: {similarity_score}")
