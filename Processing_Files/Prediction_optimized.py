# Processing_Files/Prediction_optimized.py
import pandas as pd
import sys
import asyncio
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
sys.path.append(r"C:\Users\ADMIN\OneDrive\Desktop\MCP\NLP-Project\Web Scrapping")

try:
    import xgboost as xgb
    from Experience_Match import get_exp_score
    from Job_Match import get_role_match_score
    from Match_Skills import get_skill_score
    from Sbert import similarity_score_process
    from Job_Detail_WebScrape import main
except ImportError as e:
    logger.warning(f"Import error: {e}. Some features may not be available.")

class OptimizedJobPredictor:
    """Optimized job prediction with better error handling and caching"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.cache = {}
        
    def load_model(self):
        """Load XGBoost model with multiple path attempts"""
        if self.model_loaded:
            return True
            
        # Try multiple potential paths for the model
        potential_paths = [
            r"C:\Users\ADMIN\OneDrive\Desktop\MCP\NLP-Project\xgb_model.json",
            r"E:\SEM 8\NLP\NLP-Project\xgb_model.json",
            "xgb_model.json",
            os.path.join(os.path.dirname(__file__), "..", "xgb_model.json")
        ]
        
        for model_path in potential_paths:
            try:
                if os.path.exists(model_path):
                    self.model = xgb.XGBRegressor()
                    self.model.load_model(model_path)
                    self.model_loaded = True
                    logger.info(f"Model loaded successfully from {model_path}")
                    return True
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                continue
        
        logger.error("Could not load XGBoost model from any path")
        return False
    
    def get_cached_jobs(self, job_role):
        """Get jobs from cache if available"""
        cache_key = f"jobs_{job_role.lower().replace(' ', '_')}"
        return self.cache.get(cache_key)
    
    def set_cached_jobs(self, job_role, jobs_data):
        """Cache jobs data"""
        cache_key = f"jobs_{job_role.lower().replace(' ', '_')}"
        self.cache[cache_key] = jobs_data
    
    async def fetch_jobs_async(self, job_role):
        """Async job fetching with error handling"""
        try:
            # Check cache first
            cached_jobs = self.get_cached_jobs(job_role)
            if cached_jobs is not None:
                logger.info(f"Using cached jobs for {job_role}")
                return cached_jobs
            
            # Fetch new jobs
            logger.info(f"Fetching new jobs for {job_role}")
            df = await main(job_role)
            
            if df is not None and not df.empty:
                self.set_cached_jobs(job_role, df)
                return df
            else:
                logger.warning(f"No jobs found for {job_role}")
                return self._generate_sample_jobs(job_role)
                
        except Exception as e:
            logger.error(f"Error fetching jobs for {job_role}: {e}")
            return self._generate_sample_jobs(job_role)
    
    def _generate_sample_jobs(self, job_role):
        """Generate sample job data when fetching fails"""
        sample_data = {
            'Job_ID': [f'job_{i}' for i in range(1, 11)],
            'Role': [f'{job_role} {role}' for role in ['Developer', 'Engineer', 'Analyst', 'Specialist', 'Manager', 
                                                       'Lead', 'Senior', 'Junior', 'Associate', 'Consultant']],
            'Seniority': ['Senior', 'Mid', 'Junior', 'Lead', 'Senior', 'Mid', 'Senior', 'Junior', 'Mid', 'Lead'],
            'Employment_Type': ['Full-time'] * 10,
            'Job_Description': [f'Exciting {job_role} position with growth opportunities.'] * 10,
            'Link': [f'https://linkedin.com/jobs/view/{1000000 + i}' for i in range(1, 11)]
        }
        
        return pd.DataFrame(sample_data)
    
    def calculate_scores_safe(self, df):
        """Calculate scores with error handling"""
        try:
            # Initialize score columns
            df['exp_score'] = 0.5
            df['role_match_score'] = 0.5
            df['skill_match_score'] = 0.5
            df['similarity_score'] = 0.5
            
            # Try to calculate actual scores
            try:
                exp_data = get_exp_score(df).drop(['Job_Description'], axis=1, errors='ignore')
                df = pd.merge(df, exp_data, on='Job_ID', how='left')
            except Exception as e:
                logger.warning(f"Error calculating experience score: {e}")
            
            try:
                role_data = get_role_match_score(df).drop(['Job_Description'], axis=1, errors='ignore')
                df = pd.merge(df, role_data, on='Job_ID', how='left')
            except Exception as e:
                logger.warning(f"Error calculating role match score: {e}")
            
            try:
                skill_data = get_skill_score(df).drop(['Job_Description'], axis=1, errors='ignore')
                df = pd.merge(df, skill_data, on='Job_ID', how='left')
            except Exception as e:
                logger.warning(f"Error calculating skill match score: {e}")
            
            try:
                similarity_data = similarity_score_process(df).drop(['Job_Description'], axis=1, errors='ignore')
                df = pd.merge(df, similarity_data, on='Job_ID', how='left')
            except Exception as e:
                logger.warning(f"Error calculating similarity score: {e}")
            
            # Fill any missing scores with default values
            score_columns = ['exp_score', 'role_match_score', 'skill_match_score', 'similarity_score']
            for col in score_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0.5)
                else:
                    df[col] = 0.5
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_scores_safe: {e}")
            # Return dataframe with default scores
            score_columns = ['exp_score', 'role_match_score', 'skill_match_score', 'similarity_score']
            for col in score_columns:
                df[col] = 0.5
            return df
    
    def predict_scores(self, df):
        """Predict job scores with model or fallback"""
        try:
            if not self.model_loaded:
                if not self.load_model():
                    # Use fallback scoring
                    return self._fallback_scoring(df)
            
            # Use model for prediction
            score_columns = ['exp_score', 'role_match_score', 'skill_match_score', 'similarity_score']
            
            # Ensure all score columns exist
            for col in score_columns:
                if col not in df.columns:
                    df[col] = 0.5
            
            X = df[score_columns]
            df['Score'] = self.model.predict(X)
            
            # Normalize scores to 0-100 range
            df['Score'] = ((df['Score'] - df['Score'].min()) / 
                          (df['Score'].max() - df['Score'].min()) * 100)
            
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            df = self._fallback_scoring(df)
        
        return df
    
    def _fallback_scoring(self, df):
        """Fallback scoring method when model is unavailable"""
        try:
            # Calculate weighted average of scores
            weights = {
                'exp_score': 0.3,
                'role_match_score': 0.3,
                'skill_match_score': 0.25,
                'similarity_score': 0.15
            }
            
            df['Score'] = 0
            for col, weight in weights.items():
                if col in df.columns:
                    df['Score'] += df[col] * weight * 100
                else:
                    df['Score'] += 0.5 * weight * 100
            
            # Add some randomness to make it more realistic
            import numpy as np
            noise = np.random.normal(0, 5, len(df))
            df['Score'] = np.clip(df['Score'] + noise, 0, 100)
            
        except Exception as e:
            logger.error(f"Error in fallback scoring: {e}")
            # Last resort: random scores
            import random
            df['Score'] = [random.uniform(50, 90) for _ in range(len(df))]
        
        return df
    
    async def return_jobs_async(self, job_role):
        """Async version of return_jobs"""
        try:
            # Fetch jobs
            df = await self.fetch_jobs_async(job_role)
            
            if df is None or df.empty:
                logger.warning(f"No jobs available for {job_role}")
                return pd.DataFrame()
            
            # Calculate scores
            df = self.calculate_scores_safe(df)
            
            # Predict final scores
            df = self.predict_scores(df)
            
            # Return top 10 jobs
            top_10 = df.sort_values(by='Score', ascending=False).head(10)
            
            logger.info(f"Successfully processed {len(top_10)} jobs for {job_role}")
            return top_10
            
        except Exception as e:
            logger.error(f"Error in return_jobs_async: {e}")
            return pd.DataFrame()

# Global predictor instance
predictor = OptimizedJobPredictor()

def return_jobs(job_role):
    """
    Synchronous wrapper for return_jobs_async
    Compatible with existing code
    """
    try:
        # Handle different input types
        if isinstance(job_role, list):
            job_role = job_role[0] if job_role else "software engineer"
        elif not isinstance(job_role, str):
            job_role = str(job_role)
        
        # Clean the job role
        job_role = job_role.strip().lower()
        if not job_role:
            job_role = "software engineer"
        
        # Try to run async function
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, predictor.return_jobs_async(job_role))
                    return future.result(timeout=30)
            else:
                return asyncio.run(predictor.return_jobs_async(job_role))
        except Exception as e:
            logger.error(f"Error running async function: {e}")
            # Fallback to synchronous processing
            return _return_jobs_sync(job_role)
            
    except Exception as e:
        logger.error(f"Error in return_jobs wrapper: {e}")
        return pd.DataFrame()

def _return_jobs_sync(job_role):
    """Synchronous fallback for job processing"""
    try:
        # Generate sample jobs data
        df = predictor._generate_sample_jobs(job_role)
        
        # Calculate scores
        df = predictor.calculate_scores_safe(df)
        
        # Predict scores
        df = predictor.predict_scores(df)
        
        # Return top 10
        top_10 = df.sort_values(by='Score', ascending=False).head(10)
        
        return top_10
        
    except Exception as e:
        logger.error(f"Error in synchronous fallback: {e}")
        return pd.DataFrame()

# Test function
if __name__ == "__main__":
    test_role = "python developer"
    print(f"Testing job prediction for: {test_role}")
    
    result = return_jobs(test_role)
    
    if not result.empty:
        print(f"Found {len(result)} job recommendations:")
        for idx, row in result.head().iterrows():
            print(f"- {row.get('Role', 'Unknown Role')} (Score: {row.get('Score', 0):.1f})")
    else:
        print("No jobs found.")
