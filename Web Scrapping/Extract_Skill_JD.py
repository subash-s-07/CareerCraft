import pandas as pd
import re
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import os
import time
from gensim.models.phrases import Phrases, Phraser
from concurrent.futures import ThreadPoolExecutor
import pickle

class SkillExtractor:
    def __init__(self, use_pretrained=True, use_domain_adaptation=True):
        for resource in ['punkt', 'stopwords', 'wordnet']:
            nltk.download(resource, quiet=True)
        
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['experience', 'job', 'work', 'candidate', 'position', 'required', 'requirement', 'skill', 'skills', 'year', 'years', 'ability'])

        self.model_dir = "skill_extractor_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.skills_dict_path = os.path.join(self.model_dir, "skills_dictionary.pkl")
        self.phrases_model_path = os.path.join(self.model_dir, "phrases_model.pkl")
        
        self.skills_dict = self.load_skills_dict()
        self.phraser = self.load_phraser()
        self.use_domain_adaptation = use_domain_adaptation

    def load_skills_dict(self):
        if os.path.exists(self.skills_dict_path):
            with open(self.skills_dict_path, 'rb') as f:
                return pickle.load(f)
        return {"technical": set(), "soft": {"communication", "teamwork", "leadership"}, "domain_specific": set()}

    def load_phraser(self):
        if os.path.exists(self.phrases_model_path):
            with open(self.phrases_model_path, 'rb') as f:
                return pickle.load(f)
        return None

    def train_phrases_model(self, texts):
        tokenized_texts = [[token.text.lower() for token in self.nlp(text)] for text in texts]
        phrases_model = Phrases(tokenized_texts, min_count=3, threshold=5)
        self.phraser = Phraser(phrases_model)
        with open(self.phrases_model_path, 'wb') as f:
            pickle.dump(self.phraser, f)
        return self.phraser

    def adapt_to_domain(self, job_descriptions):
        if not self.use_domain_adaptation:
            return
        
        if self.phraser is None:
            self.train_phrases_model(job_descriptions)
        
        tfidf = TfidfVectorizer(max_features=300, stop_words=list(self.stop_words), ngram_range=(1, 3))
        tfidf_matrix = tfidf.fit_transform(job_descriptions)
        
        feature_names = tfidf.get_feature_names_out()
        importance = tfidf_matrix.sum(axis=0).A1
        potential_skills = [(feature, score) for feature, score in zip(feature_names, importance) if score > 0.1]
        potential_skills.sort(key=lambda x: x[1], reverse=True)
        
        for skill, _ in potential_skills[:100]:
            if len(skill.split()) <= 3 and not any(c.isdigit() for c in skill):
                self.skills_dict["domain_specific"].add(skill)
        
        with open(self.skills_dict_path, 'wb') as f:
            pickle.dump(self.skills_dict, f)
    
    def extract_skills_from_text(self, text):
        if not text or not isinstance(text, str):
            return []
        
        text = text.lower()
        matched_skills = set()
        all_skills = {skill for category in self.skills_dict.values() for skill in category}
        for skill in all_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                matched_skills.add(skill)
        
        doc = self.nlp(text[:100000])
        noun_phrases = {chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3}
        
        if self.phraser:
            phrases = self.phraser[text.split()]
            noun_phrases.update(phrase.replace('_', ' ') for phrase in phrases if '_' in phrase)
        
        return list(matched_skills.union(noun_phrases) - self.stop_words)

    def batch_extract_skills(self, job_descriptions, use_parallel=True):
        start_time = time.time()
        if self.use_domain_adaptation:
            self.adapt_to_domain(job_descriptions)
        
        if use_parallel and len(job_descriptions) > 10:
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                all_skills = list(executor.map(self.extract_skills_from_text, job_descriptions))
        else:
            all_skills = [self.extract_skills_from_text(jd) for jd in job_descriptions]
        
        skill_occurrences = Counter(skill for skills in all_skills for skill in skills)
        print(f"Processed {len(job_descriptions)} job descriptions in {time.time() - start_time:.2f} seconds")
        
        return all_skills, skill_occurrences

    def process_job_descriptions_file(self, file_path, format='csv', text_column='description'):
        try:
            df = pd.read_csv(file_path) if format.lower() == 'csv' else pd.read_excel(file_path)
        except Exception as e:
            print(f"Error loading file: {e}")
            return None, None
        
        job_descriptions = df.get(text_column, '').fillna('').tolist()
        skills_per_job, overall_skills = self.batch_extract_skills(job_descriptions)
        df['extracted_skills'] = skills_per_job
        return df

def compute_tfidf_weighted_skills(job_descriptions, extracted_skills_per_job):
    all_skills = {skill for skills in extracted_skills_per_job for skill in skills}
    tfidf = TfidfVectorizer(vocabulary=list(all_skills), ngram_range=(1, 3))
    tfidf_matrix = tfidf.fit_transform(job_descriptions)
    feature_names = tfidf.get_feature_names_out()
    feature_index = {feature: idx for idx, feature in enumerate(feature_names)}
    
    top_skills_per_job = []
    for i, skills in enumerate(extracted_skills_per_job):
        skill_scores = {skill: tfidf_matrix[i, feature_index[skill]] for skill in skills if skill in feature_index}
        top_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        top_skills_per_job.append([skill for skill, _ in top_skills])
    return top_skills_per_job

def insert_skills(df):
    extractor = SkillExtractor()
    extracted_skills, _ = extractor.batch_extract_skills(df['Job_Des'].fillna('').tolist())
    df['Skills'] =extracted_skills
    df['Top_10_Skills'] = compute_tfidf_weighted_skills(df['Job_Des'], extracted_skills)
    return df
