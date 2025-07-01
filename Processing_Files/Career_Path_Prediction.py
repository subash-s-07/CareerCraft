# Advanced Career Path Prediction using Transformers
import torch
import json
import numpy as np
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, 
    AutoTokenizer, AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
import pandas as pd
from collections import defaultdict
import re
from datetime import datetime, timedelta

class CareerPathPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        try:
            # Primary model for career prediction
            self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            self.gpt_model.to(self.device)
            
            # Text generation pipeline for career advice
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentence transformer for similarity analysis
            self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            # Classification pipeline for career categorization
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.gpt_model = None
            self.text_generator = None
        
        # Career progression patterns
        self.career_patterns = {
            'software_engineer': {
                'junior': ['Software Engineer', 'Junior Developer', 'Associate Engineer'],
                'mid': ['Senior Software Engineer', 'Software Engineer II', 'Full Stack Developer'],
                'senior': ['Lead Software Engineer', 'Principal Engineer', 'Staff Engineer'],
                'leadership': ['Engineering Manager', 'Technical Lead', 'VP Engineering'],
                'executive': ['CTO', 'Head of Engineering', 'Chief Architect']
            },
            'data_scientist': {
                'junior': ['Data Analyst', 'Junior Data Scientist', 'Research Analyst'],
                'mid': ['Data Scientist', 'Senior Data Analyst', 'ML Engineer'],
                'senior': ['Senior Data Scientist', 'Principal Data Scientist', 'Lead ML Engineer'],
                'leadership': ['Data Science Manager', 'Head of Analytics', 'ML Team Lead'],
                'executive': ['Chief Data Officer', 'VP of Data Science', 'Head of AI']
            },
            'product_manager': {
                'junior': ['Associate Product Manager', 'Junior PM', 'Product Analyst'],
                'mid': ['Product Manager', 'Senior Product Analyst', 'Product Owner'],
                'senior': ['Senior Product Manager', 'Principal PM', 'Lead Product Manager'],
                'leadership': ['Director of Product', 'VP Product', 'Head of Product'],
                'executive': ['Chief Product Officer', 'VP of Strategy', 'Chief Strategy Officer']
            },
            'marketing': {
                'junior': ['Marketing Specialist', 'Marketing Coordinator', 'Digital Marketing Associate'],
                'mid': ['Marketing Manager', 'Senior Marketing Specialist', 'Campaign Manager'],
                'senior': ['Senior Marketing Manager', 'Marketing Director', 'Brand Manager'],
                'leadership': ['VP Marketing', 'Head of Marketing', 'Marketing Director'],
                'executive': ['Chief Marketing Officer', 'VP of Growth', 'Chief Brand Officer']
            }
        }
        
        # Industry transition patterns
        self.industry_transitions = {
            'tech': ['fintech', 'healthtech', 'edtech', 'e-commerce', 'saas'],
            'finance': ['fintech', 'consulting', 'real estate', 'insurance'],
            'healthcare': ['healthtech', 'pharmaceuticals', 'biotech', 'medical devices'],
            'consulting': ['strategy', 'operations', 'technology consulting', 'management']
        }

    def analyze_current_position(self, skills, experience_years, current_role=""):
        """Analyze current career position and level"""
        # Determine career track based on skills
        career_track = self.identify_career_track(skills)
        
        # Determine current level based on experience
        current_level = self.determine_career_level(experience_years, skills)
        
        # Get role suggestions for current level
        current_roles = self.get_roles_for_level(career_track, current_level)
        
        return {
            'career_track': career_track,
            'current_level': current_level,
            'suggested_current_roles': current_roles,
            'experience_years': experience_years,
            'primary_skills': skills[:5] if skills else []
        }

    def identify_career_track(self, skills):
        """Identify primary career track based on skills"""
        if not skills:
            return 'software_engineer'  # Default
        
        # Define skill-to-track mapping
        track_keywords = {
            'software_engineer': ['python', 'java', 'javascript', 'react', 'angular', 'node.js', 'software development', 'programming'],
            'data_scientist': ['machine learning', 'data science', 'statistics', 'tensorflow', 'pytorch', 'data analysis', 'sql', 'pandas'],
            'product_manager': ['product management', 'product strategy', 'roadmap', 'agile', 'scrum', 'user experience'],
            'marketing': ['marketing', 'digital marketing', 'seo', 'content marketing', 'social media', 'campaigns']
        }
        
        # Score each track based on skill matches
        track_scores = defaultdict(int)
        skills_lower = [skill.lower() for skill in skills]
        
        for track, keywords in track_keywords.items():
            for skill in skills_lower:
                for keyword in keywords:
                    if keyword in skill or skill in keyword:
                        track_scores[track] += 1
        
        # Return track with highest score
        if track_scores:
            return max(track_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'software_engineer'  # Default fallback

    def determine_career_level(self, experience_years, skills):
        """Determine career level based on experience and skills"""
        # Advanced skills that indicate seniority
        senior_skills = [
            'architecture', 'system design', 'leadership', 'mentoring', 'team lead',
            'principal', 'senior', 'lead', 'management', 'strategy'
        ]
        
        # Count senior skill indicators
        senior_skill_count = sum(1 for skill in skills for senior_skill in senior_skills 
                                if senior_skill.lower() in skill.lower())
        
        # Determine level based on experience and skills
        if experience_years < 2:
            return 'junior'
        elif experience_years < 5:
            return 'mid' if senior_skill_count < 2 else 'senior'
        elif experience_years < 8:
            return 'senior' if senior_skill_count < 3 else 'leadership'
        elif experience_years < 12:
            return 'leadership' if senior_skill_count < 4 else 'executive'
        else:
            return 'executive'

    def get_roles_for_level(self, career_track, level):
        """Get role suggestions for a specific career track and level"""
        if career_track in self.career_patterns and level in self.career_patterns[career_track]:
            return self.career_patterns[career_track][level]
        return ['Professional', 'Specialist', 'Expert']  # Generic fallback

    def predict_next_career_steps(self, current_analysis, time_horizon_years=3):
        """Predict next career steps using transformer models"""
        career_track = current_analysis['career_track']
        current_level = current_analysis['current_level']
        skills = current_analysis['primary_skills']
        experience = current_analysis['experience_years']
        
        # Determine next levels
        level_progression = ['junior', 'mid', 'senior', 'leadership', 'executive']
        current_index = level_progression.index(current_level) if current_level in level_progression else 1
        
        predictions = []
        
        # Predict progression within same track
        for years_ahead in [1, 2, 3, 5]:
            if years_ahead <= time_horizon_years:
                predicted_level_index = min(current_index + (years_ahead // 2), len(level_progression) - 1)
                predicted_level = level_progression[predicted_level_index]
                
                # Get roles for predicted level
                predicted_roles = self.get_roles_for_level(career_track, predicted_level)
                
                # Generate AI-powered career advice
                career_advice = self.generate_career_advice(skills, experience + years_ahead, predicted_roles[0])
                
                predictions.append({
                    'timeline': f"{years_ahead} year{'s' if years_ahead > 1 else ''}",
                    'predicted_level': predicted_level,
                    'suggested_roles': predicted_roles,
                    'required_skills': self.get_skills_for_level(career_track, predicted_level),
                    'career_advice': career_advice,
                    'confidence': self.calculate_prediction_confidence(experience, years_ahead)
                })
        
        # Predict lateral moves and industry transitions
        lateral_moves = self.predict_lateral_moves(current_analysis)
        
        return {
            'vertical_progression': predictions,
            'lateral_opportunities': lateral_moves,
            'industry_transitions': self.predict_industry_transitions(career_track),
            'skill_development_priorities': self.get_skill_development_priorities(current_analysis)
        }

    def generate_career_advice(self, skills, experience_years, target_role):
        """Generate AI-powered career advice using GPT"""
        if not self.text_generator:
            return f"To become a {target_role}, focus on developing relevant skills and gaining experience."
        
        skills_text = ', '.join(skills[:5]) if skills else 'professional skills'
        
        prompt = f"Career advice for someone with {experience_years} years experience and skills in {skills_text} who wants to become a {target_role}:"
        
        try:
            generated = self.text_generator(
                prompt,
                max_length=prompt.count(' ') + 80,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.gpt_tokenizer.eos_token_id,
                do_sample=True
            )
            
            advice = generated[0]['generated_text'].replace(prompt, '').strip()
            return advice if advice else f"Focus on building expertise in {target_role} responsibilities and developing leadership skills."
            
        except Exception as e:
            print(f"Error generating career advice: {e}")
            return f"To advance to {target_role}, focus on developing relevant technical and leadership skills."

    def get_skills_for_level(self, career_track, level):
        """Get required skills for a specific career level"""
        skill_map = {
            'software_engineer': {
                'junior': ['Programming fundamentals', 'Version control', 'Testing'],
                'mid': ['System design basics', 'Code review', 'Mentoring juniors'],
                'senior': ['Advanced system design', 'Architecture', 'Technical leadership'],
                'leadership': ['Team management', 'Strategic planning', 'Cross-functional collaboration'],
                'executive': ['Organizational strategy', 'Budget management', 'Vision setting']
            },
            'data_scientist': {
                'junior': ['Statistics', 'Python/R', 'Data visualization'],
                'mid': ['Machine learning', 'Feature engineering', 'Model deployment'],
                'senior': ['Advanced ML', 'MLOps', 'Research leadership'],
                'leadership': ['Team building', 'Strategy', 'Stakeholder management'],
                'executive': ['AI strategy', 'Business transformation', 'Innovation leadership']
            }
        }
        
        if career_track in skill_map and level in skill_map[career_track]:
            return skill_map[career_track][level]
        return ['Professional development', 'Leadership', 'Strategic thinking']

    def predict_lateral_moves(self, current_analysis):
        """Predict possible lateral career moves"""
        current_track = current_analysis['career_track']
        skills = current_analysis['primary_skills']
        
        # Define lateral move possibilities
        lateral_moves = {
            'software_engineer': ['DevOps Engineer', 'Solutions Architect', 'Technical Product Manager'],
            'data_scientist': ['ML Engineer', 'Research Scientist', 'Data Engineering'],
            'product_manager': ['Business Analyst', 'Strategy Consultant', 'Operations Manager'],
            'marketing': ['Sales', 'Business Development', 'Customer Success']
        }
        
        moves = lateral_moves.get(current_track, ['Consultant', 'Analyst', 'Specialist'])
        
        # Score moves based on skill compatibility
        scored_moves = []
        for move in moves:
            compatibility_score = self.calculate_move_compatibility(skills, move)
            scored_moves.append({
                'role': move,
                'compatibility_score': compatibility_score,
                'transition_difficulty': 'Easy' if compatibility_score > 0.7 else 'Moderate' if compatibility_score > 0.4 else 'Challenging',
                'recommended_prep_time': '3-6 months' if compatibility_score > 0.6 else '6-12 months'
            })
        
        return sorted(scored_moves, key=lambda x: x['compatibility_score'], reverse=True)

    def calculate_move_compatibility(self, skills, target_role):
        """Calculate compatibility score for a lateral move"""
        # This is a simplified version - in practice, you'd have more sophisticated matching
        role_keywords = {
            'DevOps Engineer': ['automation', 'infrastructure', 'deployment', 'monitoring'],
            'Solutions Architect': ['system design', 'architecture', 'scalability', 'integration'],
            'Technical Product Manager': ['product', 'strategy', 'roadmap', 'technical'],
            'ML Engineer': ['machine learning', 'deployment', 'production', 'mlops'],
            'Research Scientist': ['research', 'algorithms', 'publications', 'innovation']
        }
        
        if target_role not in role_keywords:
            return 0.5  # Default compatibility
        
        target_keywords = role_keywords[target_role]
        skills_lower = [skill.lower() for skill in skills]
        
        matches = sum(1 for keyword in target_keywords 
                     if any(keyword in skill for skill in skills_lower))
        
        return matches / len(target_keywords) if target_keywords else 0.5

    def predict_industry_transitions(self, current_track):
        """Predict possible industry transitions"""
        # Industry transition recommendations based on track
        transitions = {
            'software_engineer': [
                {'industry': 'FinTech', 'growth_potential': 'High', 'difficulty': 'Medium'},
                {'industry': 'HealthTech', 'growth_potential': 'Very High', 'difficulty': 'Medium'},
                {'industry': 'Gaming', 'growth_potential': 'High', 'difficulty': 'Low'},
                {'industry': 'Blockchain/Web3', 'growth_potential': 'High', 'difficulty': 'High'}
            ],
            'data_scientist': [
                {'industry': 'AI/ML Startups', 'growth_potential': 'Very High', 'difficulty': 'Low'},
                {'industry': 'Autonomous Vehicles', 'growth_potential': 'High', 'difficulty': 'High'},
                {'industry': 'Pharmaceutical', 'growth_potential': 'Medium', 'difficulty': 'Medium'},
                {'industry': 'Climate Tech', 'growth_potential': 'Very High', 'difficulty': 'Medium'}
            ]
        }
        
        return transitions.get(current_track, [
            {'industry': 'Technology', 'growth_potential': 'High', 'difficulty': 'Medium'},
            {'industry': 'Consulting', 'growth_potential': 'Medium', 'difficulty': 'Low'}
        ])

    def get_skill_development_priorities(self, current_analysis):
        """Get prioritized skill development recommendations"""
        career_track = current_analysis['career_track']
        current_level = current_analysis['current_level']
        current_skills = current_analysis['primary_skills']
        
        # Get skills needed for next level
        level_progression = ['junior', 'mid', 'senior', 'leadership', 'executive']
        current_index = level_progression.index(current_level) if current_level in level_progression else 1
        next_level = level_progression[min(current_index + 1, len(level_progression) - 1)]
        
        required_skills = self.get_skills_for_level(career_track, next_level)
        
        # Identify skill gaps
        current_skills_lower = [skill.lower() for skill in current_skills]
        skill_gaps = [skill for skill in required_skills 
                     if not any(skill.lower() in curr_skill for curr_skill in current_skills_lower)]
        
        # Prioritize skills
        priorities = []
        for i, skill in enumerate(skill_gaps[:5]):  # Top 5 priorities
            priority_level = 'High' if i < 2 else 'Medium' if i < 4 else 'Low'
            learning_resources = self.get_learning_resources(skill)
            
            priorities.append({
                'skill': skill,
                'priority': priority_level,
                'timeline': '3-6 months' if priority_level == 'High' else '6-12 months',
                'learning_resources': learning_resources
            })
        
        return priorities

    def get_learning_resources(self, skill):
        """Get learning resources for a specific skill"""
        resource_map = {
            'system design': ['System Design Interview courses', 'Designing Data-Intensive Applications book'],
            'machine learning': ['Coursera ML Course', 'Hands-On Machine Learning book'],
            'leadership': ['Leadership training programs', 'Management courses'],
            'strategic thinking': ['Strategy courses', 'Business strategy books'],
            'team management': ['Management training', 'Leadership workshops']
        }
        
        skill_lower = skill.lower()
        for key, resources in resource_map.items():
            if key in skill_lower:
                return resources
        
        return [f'Online courses for {skill}', f'Professional training in {skill}']

    def calculate_prediction_confidence(self, experience_years, years_ahead):
        """Calculate confidence level for career predictions"""
        # Higher confidence for near-term predictions and experienced professionals
        base_confidence = 0.8
        
        # Reduce confidence for longer predictions
        time_penalty = 0.1 * years_ahead
        
        # Increase confidence for more experienced professionals
        experience_bonus = min(0.2, experience_years * 0.02)
        
        confidence = base_confidence - time_penalty + experience_bonus
        return max(0.3, min(0.95, confidence))  # Keep between 30% and 95%

    def generate_comprehensive_career_plan(self, skills, experience_years, current_role="", goals=""):
        """Generate a comprehensive career development plan"""
        # Analyze current position
        current_analysis = self.analyze_current_position(skills, experience_years, current_role)
        
        # Predict career paths
        career_predictions = self.predict_next_career_steps(current_analysis)
        
        # Generate personalized recommendations
        plan = {
            'current_analysis': current_analysis,
            'career_predictions': career_predictions,
            'immediate_actions': self.get_immediate_action_items(current_analysis),
            'long_term_strategy': self.get_long_term_strategy(current_analysis, goals),
            'success_metrics': self.define_success_metrics(current_analysis),
            'generated_at': datetime.now().isoformat()
        }
        
        return plan

    def get_immediate_action_items(self, current_analysis):
        """Get immediate actionable items"""
        return [
            "Update resume with latest skills and achievements",
            "Identify and connect with professionals in target roles",
            "Start learning the top priority skill for your next career level",
            "Seek feedback from managers or mentors on career goals",
            "Join professional communities or forums in your field"
        ]

    def get_long_term_strategy(self, current_analysis, goals):
        """Generate long-term career strategy"""
        career_track = current_analysis['career_track']
        
        strategy = {
            'vision': f"Become a recognized expert and leader in {career_track.replace('_', ' ')}",
            'key_milestones': [
                "Achieve next career level within 2-3 years",
                "Develop expertise in emerging technologies",
                "Build strong professional network",
                "Gain leadership experience through projects or mentoring"
            ],
            'success_timeline': "5-7 years to reach senior leadership level"
        }
        
        if goals:
            strategy['personalized_goals'] = goals
        
        return strategy

    def define_success_metrics(self, current_analysis):
        """Define metrics to measure career success"""
        return {
            'quantitative': [
                "Salary growth of 15-25% year over year",
                "Promotion to next level within 2-3 years",
                "Completion of skill development goals"
            ],
            'qualitative': [
                "Increased responsibility and autonomy",
                "Recognition as subject matter expert",
                "Strong professional network and mentorship opportunities",
                "Work-life balance and job satisfaction"
            ]
        }

# Integration functions for the main application
def predict_career_path(resume_data):
    """Main function to integrate career path prediction with existing system"""
    predictor = CareerPathPredictor()
    
    # Extract data from resume
    skills = resume_data.get('Skills', [])
    experience_years = resume_data.get('Total Work Experience in Years', 0)
    current_role = resume_data.get('current_role', '')
    
    # Generate comprehensive career plan
    career_plan = predictor.generate_comprehensive_career_plan(
        skills, experience_years, current_role
    )
    
    return career_plan

def get_next_career_steps(skills, experience_years, time_horizon=3):
    """Simplified function to get next career steps"""
    predictor = CareerPathPredictor()
    current_analysis = predictor.analyze_current_position(skills, experience_years)
    predictions = predictor.predict_next_career_steps(current_analysis, time_horizon)
    
    return {
        'current_position': current_analysis,
        'predictions': predictions
    }

def analyze_career_progression(resume_data, target_role=""):
    """Analyze career progression towards a specific target role"""
    predictor = CareerPathPredictor()
    
    skills = resume_data.get('Skills', [])
    experience_years = resume_data.get('Total Work Experience in Years', 0)
    
    current_analysis = predictor.analyze_current_position(skills, experience_years)
    
    # If target role specified, provide specific guidance
    if target_role:
        gap_analysis = {
            'current_level': current_analysis['current_level'],
            'target_role': target_role,
            'skill_gaps': predictor.get_skills_for_level(current_analysis['career_track'], 'senior'),
            'estimated_timeline': '2-4 years',
            'recommended_actions': predictor.get_immediate_action_items(current_analysis)
        }
        return gap_analysis
    
    return current_analysis
