import json
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch
import re
from collections import defaultdict

# Enhanced models for intelligent skill analysis
skill_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

with open(r'Resume_parse.json', 'r') as file:
    resume = json.load(file)
# Enhanced skill analysis functions
def extract_skills_from_text(text):
    """Extract skills from text using advanced NLP techniques"""
    # Skill categories for zero-shot classification
    skill_categories = [
        "programming languages", "machine learning", "data science", "cloud computing",
        "web development", "mobile development", "database management", "devops",
        "project management", "business analysis", "cybersecurity", "networking",
        "quality assurance", "testing", "artificial intelligence", "data analysis"
    ]
    
    extracted_skills = set()
    
    # Method 1: Pattern-based extraction
    skill_patterns = {
        'programming': r'\b(?:Python|Java|JavaScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|TypeScript|Scala|R|MATLAB)\b',
        'frameworks': r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel|Rails|TensorFlow|PyTorch|Keras)\b',
        'databases': r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|SQL Server|SQLite|DynamoDB)\b',
        'cloud': r'\b(?:AWS|Azure|Google Cloud|GCP|Docker|Kubernetes|Jenkins|GitLab)\b',
        'tools': r'\b(?:Git|JIRA|Confluence|Tableau|Power BI|Excel|Photoshop|Figma|Slack)\b'
    }
    
    for category, pattern in skill_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        extracted_skills.update([match.strip() for match in matches])
    
    # Method 2: Zero-shot classification for skill categories
    try:
        classification_result = skill_classifier(text, skill_categories)
        for label, score in zip(classification_result['labels'], classification_result['scores']):
            if score > 0.3:  # Confidence threshold
                category_skills = extract_skills_by_category(text, label)
                extracted_skills.update(category_skills)
    except Exception as e:
        print(f"Error in skill classification: {e}")
    
    # Method 3: Context-based extraction
    context_skills = extract_contextual_skills(text)
    extracted_skills.update(context_skills)
    
    return list(extracted_skills)

def extract_skills_by_category(text, category):
    """Extract specific skills for a given category"""
    category_patterns = {
        'programming languages': r'\b(?:Python|Java|JavaScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|TypeScript)\b',
        'machine learning': r'\b(?:TensorFlow|PyTorch|Keras|Scikit-learn|XGBoost|Neural Networks|Deep Learning|NLP)\b',
        'cloud computing': r'\b(?:AWS|Azure|Google Cloud|Docker|Kubernetes|Lambda|EC2|S3)\b',
        'web development': r'\b(?:React|Angular|Vue|HTML|CSS|Node\.js|Express|Django|Flask)\b',
        'data science': r'\b(?:Pandas|NumPy|Matplotlib|Jupyter|R|Statistics|Data Mining)\b'
    }
    
    pattern = category_patterns.get(category.lower())
    if pattern:
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [match.strip() for match in matches]
    return []

def extract_contextual_skills(text):
    """Extract skills based on context using linguistic patterns"""
    skills = set()
    
    # Context patterns that indicate skills
    skill_context_patterns = [
        r'experience (?:with|in|using) ([^,.]{2,30})',
        r'skilled in ([^,.]{2,30})',
        r'proficient (?:in|with) ([^,.]{2,30})',
        r'expertise in ([^,.]{2,30})',
        r'knowledge of ([^,.]{2,30})',
        r'familiar with ([^,.]{2,30})',
        r'worked with ([^,.]{2,30})',
        r'using ([^,.]{2,30})',
        r'technologies?:? ([^,.]{2,30})'
    ]
    
    for pattern in skill_context_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            skill = match.strip()
            if is_valid_skill(skill):
                skills.add(skill)
    
    return list(skills)

def is_valid_skill(skill):
    """Validate if extracted text is actually a skill"""
    # Filter out common non-skills
    invalid_words = {
        'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'years', 'months', 'experience', 'work', 'job', 'role', 'position', 'team', 'company'
    }
    
    skill_lower = skill.lower().strip()
    
    # Check if it's just a common word
    if skill_lower in invalid_words:
        return False
    
    # Check length constraints
    if len(skill_lower) < 2 or len(skill_lower.split()) > 4:
        return False
    
    # Check if it's just numbers
    if skill_lower.isdigit():
        return False
    
    return True

def semantic_skill_matching(resume_skills, job_skills):
    """Advanced semantic skill matching using embeddings"""
    if not resume_skills or not job_skills:
        return 0.0, []
    
    # Convert skills to embeddings
    resume_embeddings = sentence_model.encode(resume_skills)
    job_embeddings = sentence_model.encode(job_skills)
    
    # Calculate similarity matrix
    similarity_matrix = util.cos_sim(resume_embeddings, job_embeddings)
    
    # Find best matches
    matches = []
    total_similarity = 0
    
    for i, job_skill in enumerate(job_skills):
        # Find best matching resume skill
        best_match_idx = similarity_matrix[:, i].argmax()
        best_similarity = similarity_matrix[best_match_idx, i].item()
        
        if best_similarity > 0.5:  # Threshold for meaningful match
            matches.append({
                'job_skill': job_skill,
                'resume_skill': resume_skills[best_match_idx],
                'similarity': best_similarity
            })
            total_similarity += best_similarity
    
    # Calculate overall match score
    match_score = total_similarity / len(job_skills) if job_skills else 0
    
    return match_score, matches

def analyze_skill_gaps(resume_skills, job_skills):
    """Analyze gaps between resume and job skills"""
    # Semantic matching to find what's missing
    match_score, matches = semantic_skill_matching(resume_skills, job_skills)
    
    # Find unmatched job skills (gaps)
    matched_job_skills = {match['job_skill'] for match in matches}
    skill_gaps = [skill for skill in job_skills if skill not in matched_job_skills]
    
    # Categorize gaps by importance
    critical_gaps = []
    nice_to_have_gaps = []
    
    for gap in skill_gaps:
        # Use zero-shot classification to determine importance
        try:
            importance_result = skill_classifier(
                f"The skill {gap} is important for this job",
                ["critical requirement", "nice to have", "not important"]
            )
            
            if importance_result['scores'][0] > 0.6:  # High confidence for critical
                critical_gaps.append(gap)
            else:
                nice_to_have_gaps.append(gap)
        except:
            nice_to_have_gaps.append(gap)  # Default to nice-to-have
    
    return {
        'match_score': match_score,
        'matches': matches,
        'critical_gaps': critical_gaps,
        'nice_to_have_gaps': nice_to_have_gaps,
        'total_gaps': len(skill_gaps),
        'gap_percentage': len(skill_gaps) / len(job_skills) if job_skills else 0
    }

def get_skill_recommendations(skill_gaps):
    """Get recommendations for filling skill gaps"""
    recommendations = []
    
    skill_learning_map = {
        'python': 'Take Python programming courses on Coursera or edX',
        'machine learning': 'Study ML fundamentals and complete hands-on projects',
        'aws': 'Get AWS certification starting with Cloud Practitioner',
        'react': 'Build React projects and complete online tutorials',
        'docker': 'Learn containerization through Docker documentation and practice',
        'kubernetes': 'Start with Kubernetes basics after mastering Docker',
        'tensorflow': 'Complete TensorFlow tutorials and build ML models',
        'sql': 'Practice SQL queries and database design',
        'git': 'Learn version control through Git tutorials and practice'
    }
    
    for gap in skill_gaps:
        gap_lower = gap.lower()
        for skill_key, recommendation in skill_learning_map.items():
            if skill_key in gap_lower:
                recommendations.append({
                    'skill': gap,
                    'recommendation': recommendation,
                    'priority': 'high' if skill_key in ['python', 'sql', 'git'] else 'medium'
                })
                break
        else:
            # Generic recommendation
            recommendations.append({
                'skill': gap,
                'recommendation': f'Find online courses or tutorials for {gap}',
                'priority': 'medium'
            })
    
    return recommendations
# Enhanced skill scoring function with backward compatibility
def get_skill_score(data):
    """Enhanced skill matching with semantic analysis"""
    hash = {
        'Job_ID': [],
        'Job_Description': [],
        'skill_match_score': [],
        'semantic_skill_score': [],
        'skill_gap_analysis': [],
        'detailed_matches': []
    }
    
    resume_skills = resume.get("Skills", [])
    if isinstance(resume_skills, str):
        resume_skills = [resume_skills]
    
    for i in range(len(data)):
        # Get job skills from the data
        job_skills = data.iloc[i]["Skills"]
        if isinstance(job_skills, str):
            # If skills are in string format, try to parse them
            if '[' in job_skills and ']' in job_skills:
                try:
                    job_skills = eval(job_skills)  # Parse list string
                except:
                    job_skills = [s.strip() for s in job_skills.replace('[', '').replace(']', '').replace("'", '').split(',')]
            else:
                job_skills = [s.strip() for s in job_skills.split(',')]
        
        # Extract additional skills from job description
        job_description = data.iloc[i]["Job_Des"]
        extracted_job_skills = extract_skills_from_text(job_description)
        
        # Combine listed skills with extracted skills
        all_job_skills = list(set(job_skills + extracted_job_skills))
        
        # Perform semantic skill matching
        semantic_score, detailed_matches = semantic_skill_matching(resume_skills, all_job_skills)
        
        # Perform gap analysis
        gap_analysis = analyze_skill_gaps(resume_skills, all_job_skills)
        
        # Calculate traditional skill match for backward compatibility
        traditional_score = compute_skill_match_traditional(resume_skills, job_skills)
        
        # Combined score (70% semantic + 30% traditional)
        combined_score = 0.7 * semantic_score + 0.3 * traditional_score
        
        hash['Job_ID'].append(data.iloc[i]['Job_ID'])
        hash['Job_Description'].append(data.iloc[i]["Job_Des"])
        hash['skill_match_score'].append(combined_score)
        hash['semantic_skill_score'].append(semantic_score)
        hash['skill_gap_analysis'].append(gap_analysis)
        hash['detailed_matches'].append(detailed_matches)
    
    return pd.DataFrame(hash)

def compute_skill_match_traditional(resume_skills, job_skills):
    """Traditional exact skill matching for backward compatibility"""
    if not resume_skills or not job_skills:
        return 0.0
    
    resume_skills_lower = [skill.lower().strip() for skill in resume_skills]
    job_skills_lower = [skill.lower().strip() for skill in job_skills]
    
    matched_skills = [skill for skill in resume_skills_lower if skill in job_skills_lower]
    return len(matched_skills) / len(resume_skills) if resume_skills else 0

def get_enhanced_skill_analysis(resume_skills, job_description):
    """Get comprehensive skill analysis for a job"""
    # Extract skills from job description
    job_skills = extract_skills_from_text(job_description)
    
    # Perform various analyses
    semantic_score, matches = semantic_skill_matching(resume_skills, job_skills)
    gap_analysis = analyze_skill_gaps(resume_skills, job_skills)
    recommendations = get_skill_recommendations(gap_analysis['critical_gaps'] + gap_analysis['nice_to_have_gaps'])
    
    return {
        'semantic_score': semantic_score,
        'skill_matches': matches,
        'gap_analysis': gap_analysis,
        'recommendations': recommendations,
        'extracted_job_skills': job_skills,
        'total_job_skills': len(job_skills),
        'total_resume_skills': len(resume_skills)
    }

# Utility functions for backward compatibility
def tolower(l):
    """Convert list to lowercase - kept for backward compatibility"""
    return [item.lower() for item in l]

def compute_skill_match(resume_skills, job_skills):
    """Original function kept for backward compatibility"""
    matched_skills = [skill for skill in resume_skills if skill in job_skills]
    print(matched_skills)
    return len(matched_skills) / len(resume_skills) if resume_skills else 0
