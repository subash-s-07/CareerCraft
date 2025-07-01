from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from Data_Preprocessing import preprocess_text

# Enhanced models for better semantic understanding
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # Upgraded from MiniLM
secondary_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')  # Backup model

# BERT for deep contextual understanding
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

# Load resume data
with open(r'Resume_parse.json', 'r') as file:
    resume = json.load(file)

# Enhanced similarity processing with multiple models and techniques
def get_bert_embedding(text, max_length=512):
    """Get BERT embedding for deeper contextual understanding"""
    inputs = bert_tokenizer(text, return_tensors='pt', max_length=max_length, 
                           truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding[0]

def enhanced_similarity_score(resume_text, job_text):
    """Calculate enhanced similarity using multiple models"""
    # Method 1: Advanced sentence transformer
    resume_emb_st = model.encode(resume_text)
    job_emb_st = model.encode(job_text)
    similarity_st = util.cos_sim(resume_emb_st, job_emb_st).item()
    
    # Method 2: Secondary model for verification
    resume_emb_st2 = secondary_model.encode(resume_text)
    job_emb_st2 = secondary_model.encode(job_text)
    similarity_st2 = util.cos_sim(resume_emb_st2, job_emb_st2).item()
    
    # Method 3: BERT for deep contextual similarity
    try:
        resume_emb_bert = get_bert_embedding(resume_text)
        job_emb_bert = get_bert_embedding(job_text)
        similarity_bert = cosine_similarity([resume_emb_bert], [job_emb_bert])[0][0]
    except Exception as e:
        print(f"BERT similarity calculation failed: {e}")
        similarity_bert = similarity_st  # Fallback
    
    # Weighted combination of similarities
    final_similarity = (
        0.5 * similarity_st +      # Primary model (50%)
        0.3 * similarity_st2 +     # Secondary model (30%)
        0.2 * similarity_bert      # BERT contextual (20%)
    )
    
    return final_similarity, {
        'sentence_transformer': similarity_st,
        'secondary_model': similarity_st2,
        'bert_contextual': similarity_bert,
        'combined_score': final_similarity
    }

def extract_job_skills(job_description):
    """Extract skills from job description using advanced NLP"""
    skill_patterns = [
        r'\b(?:Python|Java|JavaScript|React|Angular|AWS|Azure|Docker|Kubernetes|SQL|NoSQL|TensorFlow|PyTorch|Machine Learning|Data Science|AI)\b',
        r'experience (?:with|in) ([^,.]+)',
        r'knowledge of ([^,.]+)',
        r'proficient in ([^,.]+)',
        r'skills?: ([^,.]+)',
        r'familiar with ([^,.]+)',
        r'expertise in ([^,.]+)'
    ]
    
    skills = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, job_description, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str):
                skills.add(match.strip())
            elif isinstance(match, tuple):
                skills.add(match[0].strip())
    
    return list(skills)

def calculate_skill_overlap_score(resume_skills, job_skills):
    """Calculate skill overlap using semantic similarity"""
    if not resume_skills or not job_skills:
        return 0.0
    
    # Convert skills to embeddings
    resume_skill_embs = model.encode(resume_skills)
    job_skill_embs = model.encode(job_skills)
    
    # Calculate cross-similarity matrix
    similarity_matrix = util.cos_sim(resume_skill_embs, job_skill_embs)
    
    # Find best matches for each job skill
    max_similarities = similarity_matrix.max(dim=0)[0]
    skill_overlap_score = max_similarities.mean().item()
    
    return skill_overlap_score

def calculate_contextual_relevance(resume_text, job_text):
    """Calculate how contextually relevant the resume is to the job"""
    # Split into key sections
    resume_sections = {
        'skills': extract_section(resume_text, ['skill', 'technical', 'competenc']),
        'experience': extract_section(resume_text, ['experience', 'work', 'employ']),
        'education': extract_section(resume_text, ['education', 'degree', 'university'])
    }
    
    job_sections = {
        'requirements': extract_section(job_text, ['requirement', 'qualif', 'must have']),
        'responsibilities': extract_section(job_text, ['responsib', 'duties', 'role']),
        'preferred': extract_section(job_text, ['prefer', 'nice to have', 'bonus'])
    }
    
    relevance_scores = {}
    
    for resume_key, resume_section in resume_sections.items():
        for job_key, job_section in job_sections.items():
            if resume_section and job_section:
                score = util.cos_sim(
                    model.encode(resume_section),
                    model.encode(job_section)
                ).item()
                relevance_scores[f"{resume_key}_to_{job_key}"] = score
    
    return relevance_scores

def extract_section(text, keywords):
    """Extract text sections based on keywords"""
    sentences = text.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            relevant_sentences.append(sentence.strip())
    
    return ' '.join(relevant_sentences) if relevant_sentences else ''

def similarity_score_process(data):
    """Enhanced similarity processing with multiple scoring methods"""
    resume_text = preprocess_text(resume.get('summary_paragraph', ''))
    resume_skills = resume.get('Skills', [])
    
    hash = {
        'Job_ID': [],
        'Job_Description': [],
        'similarity_score': [],
        'detailed_scores': [],
        'skill_overlap_score': [],
        'combined_final_score': []
    }
    
    for i in range(len(data)):
        job_description = preprocess_text(data.iloc[i]["Job_Des"])
        
        # Calculate enhanced similarity
        similarity_score, detailed_scores = enhanced_similarity_score(resume_text, job_description)
        
        # Extract job skills and calculate overlap
        job_skills = extract_job_skills(data.iloc[i]["Job_Des"])
        skill_overlap = calculate_skill_overlap_score(resume_skills, job_skills)
        
        # Combined final score (70% semantic similarity + 30% skill overlap)
        combined_score = 0.7 * similarity_score + 0.3 * skill_overlap
        
        hash['Job_ID'].append(data.iloc[i]['Job_ID'])
        hash['Job_Description'].append(data.iloc[i]["Job_Des"])
        hash['similarity_score'].append(combined_score)  # Use combined score as main score
        hash['detailed_scores'].append(detailed_scores)
        hash['skill_overlap_score'].append(skill_overlap)
        hash['combined_final_score'].append(combined_score)
    
    return pd.DataFrame(hash)

def get_similarity_insights(resume_text, job_text):
    """Get detailed insights about the similarity calculation"""
    insights = {
        'overall_similarity': 0,
        'skill_match': 0,
        'experience_relevance': 0,
        'contextual_relevance': {},
        'recommendations': []
    }
    
    # Calculate overall similarity
    similarity_score, detailed_scores = enhanced_similarity_score(resume_text, job_text)
    insights['overall_similarity'] = similarity_score
    insights['detailed_scores'] = detailed_scores
    
    # Calculate contextual relevance
    insights['contextual_relevance'] = calculate_contextual_relevance(resume_text, job_text)
    
    # Generate recommendations based on scores
    if similarity_score < 0.3:
        insights['recommendations'].append("Consider highlighting more relevant skills and experience")
    elif similarity_score < 0.6:
        insights['recommendations'].append("Good match - consider emphasizing specific technical skills")
    else:
        insights['recommendations'].append("Excellent match - you're well-qualified for this position")
    
    return insights

# Backward compatibility function
def similarity_score_process_simple(data):
    """Simple version for backward compatibility"""
    result_df = similarity_score_process(data)
    return result_df[['Job_ID', 'Job_Description', 'similarity_score']]
