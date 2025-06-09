# app_fixed.py
import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
import PyPDF2
from io import BytesIO
import docx2txt
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import re
import sys
import asyncio
sys.path.append(r"C:\Users\ADMIN\OneDrive\Desktop\MCP\NLP-Project\Processing_Files")
from Prediction import return_jobs

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Firebase initialization
import firebase_admin
from firebase_admin import credentials, firestore

# Check if Firebase app is already initialized to avoid re-initialization
if not firebase_admin._apps:
    cred = credentials.Certificate("static/nlp-project-d88b9-firebase-adminsdk-fbsvc-3164e3ecb3.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Gemini API with your API key
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
qdrant_client = QdrantClient(
)

# Vector size for embeddings
vector_size = embedding_model.get_sentence_embedding_dimension()

# Collection names
resume_collection = "resumes"
course_collection = "courses1"

# Create collections if they don't exist with proper indexing (CRITICAL FIX)
def create_collection_if_not_exists(collection_name):
    try:
        qdrant_client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists")
    except Exception:
        print(f"Creating collection {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        
        # CRITICAL FIX: Create index for resume_id field to avoid filtering errors
        if collection_name == resume_collection:
            try:
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="resume_id",
                    field_schema=models.KeywordIndexParams()
                )
                print(f"Created index for resume_id in {collection_name}")
            except Exception as e:
                print(f"Warning: Could not create index for resume_id: {e}")

# Create collections
create_collection_if_not_exists(resume_collection)
create_collection_if_not_exists(course_collection)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    text = docx2txt.process(docx_file)
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def parse_resume(text):
    """
    Parse resume text to extract structured information
    Returns a structured dictionary with resume components
    """
    # Use Gemini AI to parse the resume text into structured data
    prompt = f"""
Extract the following details separately from the resume text:
1. Name
2. Contact Information:
   - Email
   - Phone
3. Summary
4. Education:
   - Degree
   - University
   - Year
5. Work Experience:
   - Company
   - Role
   - Years
6. Total Work Experience in Years(Only number)
7. Skills: **CRITICAL** - Prioritize DOMAIN/FIELD skills first (like Data Science, Machine Learning, Artificial Intelligence, Data Analysis, Business Intelligence, Software Engineering, Web Development, Cloud Computing, DevOps, Cybersecurity, etc.) BEFORE listing programming languages (Python, Java, etc.). The first skill should be the candidate's PRIMARY DOMAIN/SPECIALIZATION.
8. Tools
9. Certifications
10. Projects
11. Domain_Skills: Extract ONLY the domain/field skills (Data Science, Machine Learning, AI, Data Analysis, Business Intelligence, etc.) - NO programming languages
12. summary_paragraph: Additionally, generate a consolidated paragraph summarizing the candidate's background, skills, and experience. This paragraph should be used for similarity measurement between the resume and job description.

Provide the output in structured JSON format.

Resume Text:
{text}
"""
    try:
        response = model.generate_content(prompt)
        print(response.text)
        # Try to extract JSON from the response
        with open("Resume_parse1.json", "w") as file:
            json.dump(response.text, file, indent=4)
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_text:
            parsed_resume = json.loads(json_text.group())
        else:
            # If can't extract JSON, use the whole response as structured output
            parsed_resume = {"parsed_content": response.text}
    except Exception as e:
        print(f"Error parsing resume: {e}")
        # Fallback to simplified parsing
        parsed_resume = {
            "error": "Automatic parsing failed",
            "raw_text": text
        }
    return parsed_resume

def generate_job_descriptions(parsed_resume):
    """
    Generate relevant job descriptions using MULTIPLE SKILLS with XGBoost
    """
    print('🔍 Generating jobs using multiple skills with ML pipeline...')
    
    # Extract MULTIPLE DOMAIN skills for job matching (ENHANCED VERSION)
    domain_skills = []
    all_skills = []
    
    if isinstance(parsed_resume, dict):
        # First try to get Domain_Skills (the new field)
        if 'Domain_Skills' in parsed_resume:
            if isinstance(parsed_resume['Domain_Skills'], list):
                domain_skills = parsed_resume['Domain_Skills'][:3]  # Take top 3
            elif isinstance(parsed_resume['Domain_Skills'], str):
                domain_skills = [parsed_resume['Domain_Skills']]
        
        # If no Domain_Skills, fallback to regular Skills but filter for domain skills
        if not domain_skills and 'Skills' in parsed_resume:
            domain_keywords = ['data science', 'machine learning', 'artificial intelligence', 'ai', 'data analysis', 
                             'business intelligence', 'software engineering', 'web development', 'cloud computing', 
                             'devops', 'cybersecurity', 'data engineering', 'software development', 'backend development',
                             'frontend development', 'full stack', 'mobile development', 'game development', 
                             'product management', 'project management', 'business analyst', 'data analyst']
            
            if isinstance(parsed_resume['Skills'], list):
                # Filter skills to prioritize domain skills
                filtered_domain_skills = [skill for skill in parsed_resume['Skills'] 
                                         if any(keyword in skill.lower() for keyword in domain_keywords)]
                if filtered_domain_skills:
                    domain_skills = filtered_domain_skills[:3]  # Take top 3 domain skills
                else:
                    domain_skills = parsed_resume['Skills'][:3]  # Fallback to first 3 skills
                    
                all_skills = parsed_resume['Skills'][:5]  # Keep all skills for backup
            elif isinstance(parsed_resume['Skills'], dict):
                for category, category_skills in parsed_resume['Skills'].items():
                    if isinstance(category_skills, list):
                        all_skills.extend(category_skills)
                    else:
                        all_skills.append(f"{category}: {category_skills}")
                domain_skills = all_skills[:3]
            else:
                domain_skills = [parsed_resume['Skills']]
                all_skills = [parsed_resume['Skills']]
    
    # Ensure we have at least some skills
    if not domain_skills:
        domain_skills = ["software development", "data analysis"]
    if not all_skills:
        all_skills = domain_skills
    
    print(f"🎯 Using domain skills: {domain_skills}")
    print(f"📊 All skills available: {all_skills[:5]}")
    
    # USE MULTIPLE SKILLS: Try each domain skill and collect more jobs
    all_jobs = []
    skills_tried = []
    
    # Try each domain skill to get diverse job recommendations
    for i, skill in enumerate(domain_skills[:3]):  # Try top 3 domain skills
        try:
            print(f"\n🔄 Attempt {i+1}: Getting jobs for skill: '{skill}'")
            jobs_df = return_jobs(skill)  # YOUR ORIGINAL FUNCTION
            skills_tried.append(skill)
            
            if hasattr(jobs_df, 'to_dict'):
                # Convert DataFrame to list of dictionaries
                skill_jobs = jobs_df.head(8).to_dict('records')  # Get 8 jobs per skill
                print(f"✅ Found {len(skill_jobs)} jobs for '{skill}'")
                
                # Add skill context to each job
                for job in skill_jobs:
                    job['matched_skill'] = skill
                    job['skill_relevance'] = f"Matched based on: {skill}"
                
                all_jobs.extend(skill_jobs)
            else:
                jobs_list = jobs_df if isinstance(jobs_df, list) else []
                all_jobs.extend(jobs_list)
                
        except Exception as e:
            print(f"⚠️ Error getting jobs for skill '{skill}': {e}")
            continue
    
    # If we have jobs from multiple skills, remove duplicates and take top performers
    if all_jobs:
        # Remove duplicate jobs based on Role and Company
        seen_jobs = set()
        unique_jobs = []
        
        for job in all_jobs:
            job_key = (job.get('Role', ''), job.get('Company', ''), job.get('Link', ''))
            if job_key not in seen_jobs:
                seen_jobs.add(job_key)
                unique_jobs.append(job)
        
        # Sort by score and take top 10
        unique_jobs.sort(key=lambda x: x.get('Score', 0), reverse=True)
        top_jobs = unique_jobs[:10]  # Take top 10 unique jobs
        
        print(f"🎯 Selected {len(top_jobs)} unique top-scoring jobs from {len(all_jobs)} total")
        
        # Format the results properly
        formatted_jobs = []
        for i, job in enumerate(top_jobs):
            formatted_job = {
                "Job Title": job.get('Role', f"Job {i+1}"),
                "Company": job.get('Company', 'Company'),
                "Summary": f"Role: {job.get('Role', 'N/A')}, Seniority: {job.get('Seniority', 'N/A')}, Type: {job.get('Employment_Type', 'N/A')}",
                "Required Skills": [job.get('matched_skill', 'N/A')],
                "Responsibilities": f"Key responsibilities for {job.get('Role', 'this role')} at {job.get('Seniority', 'this level')}",
                "Qualifications": f"Required qualifications for {job.get('Seniority', 'this level')} {job.get('Role', 'position')}",
                "Type": job.get('Employment_Type', 'Full-time'),
                "Seniority": job.get('Seniority', 'Mid-level'),
                "Relevance Score": round(job.get('Score', 0), 1),
                "Link": job.get('Link', ''),
                "Matched Skill": job.get('matched_skill', ''),
                "Location": job.get('Location', 'Various')
            }
            formatted_jobs.append(formatted_job)
        
        print(f"✅ Formatted {len(formatted_jobs)} jobs using skills: {skills_tried}")
        return formatted_jobs
    
    # Fallback: Use Gemini to generate job suggestions based on multiple skills
    print("🤖 Using Gemini fallback for job generation with multiple skills...")
    skills_text = ', '.join(domain_skills[:3])
    prompt = f"""
    Generate 10 relevant job suggestions for someone with these MULTIPLE skills: {skills_text}
    
    Consider the candidate has expertise in: {', '.join(all_skills[:5])}

    For each job, provide:
    1. Job Title
    2. Company (realistic company name)
    3. Summary
    4. Required Skills (as a list)
    5. Responsibilities
    6. Qualifications
    7. Type (Full-time/Remote/Hybrid)
    8. Seniority Level
    9. Relevance Score (0-100)
    10. Location
    
    Focus on jobs that match the PRIMARY skills: {skills_text}
    Format the result as a JSON array.
    """
    
    try:
        response = model.generate_content(prompt)
        print(response.text)
        # Try to extract JSON from the response
        json_text = re.search(r'\[.*\]', response.text, re.DOTALL)
        if json_text:
            job_descriptions = json.loads(json_text.group())
        else:
            # If can't extract JSON, parse the text manually
            job_descriptions = [{
                "Job Title": f"Opportunity in {skill}",
                "Summary": f"Great opportunity for someone with {skill} expertise",
                "Required Skills": [skill],
                "Type": "Full-time",
                "Relevance Score": 85
            } for skill in domain_skills[:5]]
    except Exception as e:
        print(f"⚠️ Error generating job descriptions: {e}")
        job_descriptions = [{
            "Job Title": f"Career Opportunity in {skill}",
            "Summary": f"Excellent career opportunity for {skill} professionals",
            "Required Skills": [skill],
            "Type": "Full-time",
            "Relevance Score": 80
        } for skill in domain_skills[:3]]
    
    print(f"🎯 Generated {len(job_descriptions)} fallback jobs")
    return job_descriptions

def analyze_skill_gaps(parsed_resume, job_descriptions):
    """
    Analyze the gap between resume skills and job requirements
    Returns a list of missing skills and other gaps
    """
    print('Skill Analysis---------------------------------------------')
    print()
    # Extract skills from resume
    resume_skills = []
    print('Skills',parsed_resume)
    if isinstance(parsed_resume, dict) and 'Skills' in parsed_resume:
        if isinstance(parsed_resume['Skills'], list):
            resume_skills = parsed_resume['Skills']
        elif isinstance(parsed_resume['Skills'], dict):
            for category, skills in parsed_resume['Skills'].items():
                if isinstance(skills, list):
                    resume_skills.extend(skills)
                else:
                    resume_skills.append(skills)
        else:
            resume_skills = [parsed_resume['Skills']]
    print('Skills',resume_skills)
    # Prepare a detailed prompt for Gemini AI
    job_desc_text = json.dumps(job_descriptions[:3])  # Limit to first 3 jobs to avoid token limits
    resume_skills_text = json.dumps(resume_skills)
    print(job_desc_text)
    prompt = f"""
    Analyze the gap between the candidate's skills and the requirements in the job descriptions.
    
    Candidate's skills:
    {resume_skills_text}
    
    Job descriptions (sample of 3):
    {job_desc_text}
    
    Please identify:
    1. Missing technical skills
    2. Missing soft skills
    3. Experience gaps
    4. Education gaps
    5. Certification gaps
    
    Format your response as a JSON object with these categories as keys, and arrays of specific gaps as values.
    """
    
    try:
        response = model.generate_content(prompt)
        # Try to extract JSON from the response
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_text:
            gaps_analysis = json.loads(json_text.group())
        else:
            # If can't extract JSON, use the text as analysis
            gaps_analysis = {"analysis": response.text}
    except Exception as e:
        print(f"Error analyzing skill gaps: {e}")
        gaps_analysis = {"error": "Could not analyze skill gaps"}
    
    return gaps_analysis

# Import your original course scraping
try:
    from course_scrap import scrape_coursera_courses
    print("Using your original course scraper")
except ImportError:
    try:
        from course_scrap_lightweight import scrape_coursera_courses
        print("Using lightweight course scraper")
    except ImportError:
        def scrape_coursera_courses(query, max_courses=10):
            print("No course scraper available, using samples")
            return [
                {
                    "id": f"course-{query}-1",
                    "title": f"Advanced {query} Course",
                    "description": f"Master {query} skills",
                    "skills": [query, "Problem Solving"],
                    "level": "Intermediate"
                }
            ]

def index_courses(courses):
    """
    Index courses scraped from Coursera into the Qdrant database
    """
    if not courses:
        print("⚠️ No courses to index")
        return
    
    try:
        search_result = qdrant_client.scroll(
            collection_name=course_collection,
            limit=1
        )
        
        # Only skip if we have existing courses AND we're trying to add similar courses
        if len(search_result[0]) > 0:
            print(f"📚 Found {len(search_result[0])} existing courses, checking for duplicates...")
            # You can add duplicate checking logic here if needed
        
        # Index each course with proper error handling
        indexed_count = 0
        for i, course in enumerate(courses):
            try:
                # Ensure course has required fields
                if not course.get('title') or not course.get('id'):
                    print(f"⚠️ Skipping invalid course {i+1}: missing title or id")
                    continue
                
                # Create a comprehensive text representation of the course
                skills_text = ', '.join(course.get('skills', []))
                course_text = f"{course['title']}. {course['description']} Skills: {skills_text}. Level: {course.get('level', 'Intermediate')}"
                print(f"📖 Indexing: {course['title']} ({course.get('level', 'N/A')})")
                
                # Generate embedding
                embedding = embedding_model.encode(course_text).tolist()
                
                # Generate a valid UUID for the point ID
                point_id = str(uuid.uuid4())
                
                # Store in Qdrant
                qdrant_client.upsert(
                    collection_name=course_collection,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "original_id": course["id"],
                                **course  # Include all other course data
                            }
                        )
                    ]
                )
                indexed_count += 1
                
            except Exception as e:
                print(f"❌ Error indexing course {i+1}: {e}")
                continue
        
        print(f"✅ Successfully indexed {indexed_count}/{len(courses)} courses in Qdrant")
        
    except Exception as e:
        print(f"❌ Critical error in index_courses: {e}")
        # Don't crash the whole application if course indexing fails

def recommend_courses(skill_gaps, top_k=8):
    """
    Recommend courses to fill the identified skill gaps - ENHANCED WITH MULTIPLE SKILLS
    """
    print("📚 Recommending courses based on missing skills...")
    
    # Extract missing skills from the gaps analysis (ENHANCED)
    missing_skills = []
    
    # Process different formats of skill_gaps
    if isinstance(skill_gaps, dict):
        # Look for various categories of missing skills
        skill_categories = ['Missing technical skills', 'missing_technical_skills', 'technical_skills', 
                          'Missing soft skills', 'missing_soft_skills', 'soft_skills',
                          'Missing skills', 'missing_skills', 'skills_needed']
        
        for category in skill_categories:
            if category in skill_gaps:
                gaps = skill_gaps[category]
                if isinstance(gaps, list):
                    missing_skills.extend(gaps[:2])  # Take top 2 from each category
                elif isinstance(gaps, str):
                    missing_skills.append(gaps)
    
    if not missing_skills and "analysis" in skill_gaps:
        # Extract skills from text analysis using AI
        analysis_text = skill_gaps["analysis"]
        prompt = f"""
        Extract a list of specific technical and professional skills mentioned as missing or needed in the following text:
        
        {analysis_text}
        
        Focus on:
        - Technical skills (programming languages, tools, frameworks)
        - Professional skills (project management, data analysis, etc.)
        - Domain expertise (machine learning, cloud computing, etc.)
        
        Format the response as a JSON array of skill strings.
        Limit to 5 most important skills.
        """
        
        try:
            response = model.generate_content(prompt)
            json_text = re.search(r'\[.*\]', response.text, re.DOTALL)
            if json_text:
                extracted_skills = json.loads(json_text.group())
                missing_skills.extend(extracted_skills[:3])
            else:
                # Extract skills using regex if JSON parsing fails
                skills_matches = re.findall(r'(?:- |• |\d\. )([\w\s]+)', response.text)
                missing_skills = [skill.strip() for skill in skills_matches if skill.strip()][:3]
        except Exception as e:
            print(f"⚠️ Error extracting skills from analysis: {e}")
    
    # If still no missing skills, use smart defaults based on common career paths
    if not missing_skills:
        missing_skills = ["Python programming", "data analysis", "project management", "cloud computing"]
    
    # Clean and deduplicate skills
    cleaned_skills = list(set([skill.strip() for skill in missing_skills if skill.strip()]))
    print(f"🎯 Looking for courses for missing skills: {cleaned_skills}")
    
    # Get courses for MULTIPLE missing skills (Enhanced approach)
    all_courses = []
    skills_processed = []
    
    for i, query in enumerate(cleaned_skills[:3]):  # Process top 3 skills
        print(f"\n📖 Scraping courses for skill {i+1}: '{query}'")
        try:
            courses = scrape_coursera_courses(query=query, max_courses=6)  # Get 6 courses per skill
            if courses:
                # Add skill context to each course
                for course in courses:
                    course['target_skill'] = query
                    course['skill_category'] = f"Addresses: {query}"
                
                all_courses.extend(courses)
                skills_processed.append(query)
                print(f"✅ Found {len(courses)} courses for '{query}'")
            else:
                print(f"⚠️ No courses found for '{query}'")
        except Exception as e:
            print(f"❌ Error scraping courses for '{query}': {e}")
            continue
    
    print(f"\n📚 Total courses collected: {len(all_courses)} from skills: {skills_processed}")
    
    # Index all courses
    if all_courses:
        index_courses(all_courses)
    
    # If we have multiple skills, create a combined query for better semantic search
    if len(cleaned_skills) > 1:
        combined_query = ", ".join(cleaned_skills[:3])
        print(f"🔍 Performing semantic search for: '{combined_query}'")
    else:
        combined_query = cleaned_skills[0] if cleaned_skills else "professional development"
    
    # Generate embedding for the combined query
    try:
        query_embedding = embedding_model.encode(combined_query).tolist()
        
        # Search for relevant courses with higher limit
        search_result = qdrant_client.search(
            collection_name=course_collection,
            query_vector=query_embedding,
            limit=top_k * 2  # Search more to have options
        )
        
        # Extract course information from search results
        recommended_courses = []
        seen_courses = set()  # To avoid duplicates
        
        for result in search_result:
            course_data = result.payload
            course_title = course_data.get('title', 'Unknown')
            
            # Skip duplicates
            if course_title in seen_courses:
                continue
            seen_courses.add(course_title)
            
            # Ensure course data has all required fields for frontend display
            if 'title' not in course_data:
                course_data['title'] = course_data.get('original_id', 'Course Title')
            if 'description' not in course_data:
                course_data['description'] = 'Comprehensive course to develop your skills'
            if 'skills' not in course_data:
                course_data['skills'] = cleaned_skills[:2]  # Use our missing skills
            if 'level' not in course_data:
                course_data['level'] = 'Intermediate'
            if 'url' not in course_data:
                course_data['url'] = 'https://www.coursera.org'
            if 'partner' not in course_data:
                course_data['partner'] = 'Top University'
            
            # Add relevance information
            course_data['relevance_score'] = round(result.score * 100, 1)
            course_data['addresses_skills'] = cleaned_skills[:2]
            
            recommended_courses.append(course_data)
            
            if len(recommended_courses) >= top_k:
                break
        
        print(f"✅ Recommended {len(recommended_courses)} courses")
        for i, course in enumerate(recommended_courses):
            print(f"  {i+1}. {course.get('title', 'N/A')} - {course.get('level', 'N/A')} ({course.get('relevance_score', 0)}% match)")
        
        return recommended_courses
        
    except Exception as e:
        print(f"❌ Error in course recommendation: {e}")
        # Return a fallback list of courses
        fallback_courses = []
        for i, skill in enumerate(cleaned_skills[:3]):
            fallback_courses.append({
                'title': f'Complete {skill} Course',
                'description': f'Master {skill} with this comprehensive course',
                'skills': [skill],
                'level': 'Intermediate',
                'url': 'https://www.coursera.org',
                'partner': 'Professional Learning',
                'relevance_score': 85,
                'target_skill': skill
            })
        return fallback_courses

def index_resume(resume_id, text, parsed_resume=None):
    """Index resume chunks and parsed data in Qdrant"""
    chunks = chunk_text(text)
    
    # Generate embeddings for chunks
    for i, chunk in enumerate(chunks):
        # Get embedding for the chunk
        embedding = embedding_model.encode(chunk).tolist()
        
        # Generate a valid UUID for the point ID
        point_id = str(uuid.uuid4())
        
        # Store in Qdrant
        qdrant_client.upsert(
            collection_name=resume_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "resume_id": resume_id,
                        "chunk_id": i,
                        "text": chunk,
                        "chunk_count": len(chunks),
                        "parsed_data": parsed_resume if i == 0 else None  # Store parsed data with first chunk only
                    }
                )
            ]
        )
    
    return len(chunks)

def search_resume(query, resume_id, top_k=3):
    """Search for relevant chunks in the resume"""
    query_embedding = embedding_model.encode(query).tolist()
    
    # Search in Qdrant for relevant chunks from the specific resume
    search_result = qdrant_client.search(
        collection_name=resume_collection,
        query_vector=query_embedding,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="resume_id",
                    match=models.MatchValue(value=resume_id)
                )
            ]
        ),
        limit=top_k
    )
    
    context = ""
    for result in search_result:
        context += result.payload["text"] + "\n\n"
    
    return context

def get_parsed_resume_data(resume_id):
    """Get the parsed resume data from Qdrant"""
    # Search for the first chunk which contains the parsed data
    search_result = qdrant_client.scroll(
        collection_name=resume_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="resume_id",
                    match=models.MatchValue(value=resume_id)
                ),
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchValue(value=0)
                )
            ]
        ),
        limit=1
    )
    
    if search_result[0] and len(search_result[0]) > 0:
        return search_result[0][0].payload.get("parsed_data")
    
    return None

def get_gemini_response(query, context, resume_id, chat_id):
    chats = get_user_chats()
    
    if chat_id not in chats:
        return "Sorry, I couldn't find the chat data."
    
    chat_data = chats[chat_id]
    analysis = chat_data.get('analysis', {})
    jobs = analysis.get('job_descriptions', [])
    print(analysis)
    print("JOBS________________________")
    print(jobs)
    """Get response from Gemini model with RAG context"""
    prompt = f"""
    You are an expert resume analyzer AI assistant. Use the following context from the resume to answer the user's question. 
    If the information is not in the context, say you don't have that specific information from the resume.
    
    RESUME CONTEXT:
    {context}
    ANALYSIS:
    {analysis}
    JOBS_RECOMMENDED:
    {jobs}
    
    USER QUESTION: {query}
    
    Provide a helpful, professional response based only on the resume information provided.
    """
    
    response = model.generate_content(prompt)
    return response.text

def get_user_id():
    """Get the constant user ID - same for all sessions"""
    # Using a fixed user ID for all sessions
    return "default_user_constant_id"

def get_user_chats():
    """Retrieve user's chats from Firebase"""
    user_id = get_user_id()
    
    chats = {}
    try:
        # Get reference to the chats collection for this user
        chat_refs = db.collection('users').document(user_id).collection('chats').stream()
        
        for chat_doc in chat_refs:
            chat_data = chat_doc.to_dict()
            # Add chat_id to the chat data for easy reference
            chats[chat_doc.id] = chat_data
    except Exception as e:
        print(f"Error retrieving chats: {e}")
    
    return chats

def save_chat(chat_id, chat_data):
    """Save chat to Firebase"""
    user_id = get_user_id()
    
    try:
        # Make sure chat_data is serializable
        # Convert timestamp objects to strings if needed
        serializable_data = chat_data.copy()
        
        # Save to Firebase
        db.collection('users').document(user_id).collection('chats').document(chat_id).set(
            serializable_data, merge=True
        )
        return True
    except Exception as e:
        print(f"Error saving chat: {e}")
        return False

def delete_chat_from_firebase(chat_id):
    """Delete chat from Firebase"""
    user_id = get_user_id()
    
    try:
        db.collection('users').document(user_id).collection('chats').document(chat_id).delete()
        return True
    except Exception as e:
        print(f"Error deleting chat: {e}")
        return False

@app.route('/')
def index():
    # Get chats from Firebase using the constant user ID
    chats = get_user_chats()
    
    # Sort chats by last_accessed (most recent first)
    sorted_chats = dict(sorted(
        chats.items(), 
        key=lambda item: item[1].get('last_accessed', 0), 
        reverse=True
    ))
    
    return render_template('index.html', chats=sorted_chats)

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Generate a unique ID for the resume
        resume_id = str(uuid.uuid4())
        
        # Extract text based on file type
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif file.filename.lower().endswith(('.docx', '.doc')):
            # Create a BytesIO object for docx processing
            file_stream = BytesIO(file.read())
            text = extract_text_from_docx(file_stream)
        else:
            return jsonify({"error": "Unsupported file format. Please upload PDF or DOCX"}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{resume_id}_{filename}")
        
        # If we've already read the file for docx, we need to save it differently
        if file.filename.lower().endswith(('.docx', '.doc')):
            with open(file_path, 'wb') as f:
                f.write(file_stream.getvalue())
        else:
            file.save(file_path)
        
        # Parse the resume
        parsed_resume = parse_resume(text)
        
        # Generate job descriptions using YOUR ORIGINAL LOGIC
        job_descriptions = generate_job_descriptions(parsed_resume)
        
        # Analyze skill gaps
        skill_gaps = analyze_skill_gaps(parsed_resume, job_descriptions)
        
        # Recommend courses based on ACTUAL missing skills
        recommended_courses = recommend_courses(skill_gaps)
        
        # Index the resume in Qdrant (with parsed data)
        chunks_count = index_resume(resume_id, text, parsed_resume)
        
        # Create a new chat for this resume
        chat_id = str(uuid.uuid4())
        chat_name = f"Resume: {filename}"
        
        # Create chat data with analysis results
        chat_data = {
            'name': chat_name,
            'resume_id': resume_id,
            'filename': filename,
            'messages': [],
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_accessed': datetime.now().timestamp(),
            'parsed_resume': parsed_resume,
            'analysis': {
                'job_descriptions': job_descriptions,
                'skill_gaps': skill_gaps,
                'recommended_courses': recommended_courses,
                'last_analyzed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Save to Firebase
        if save_chat(chat_id, chat_data):
            return jsonify({
                "success": True, 
                "message": f"Resume uploaded and analyzed", 
                "chat_id": chat_id
            })
        else:
            return jsonify({"error": "Failed to save chat data"}), 500

@app.route('/analysis_qa', methods=['POST'])
def analysis_qa():
    data = request.json
    chat_id = data.get('chat_id')
    question = data.get('question')
    analysis_type = data.get('analysis_type')  # 'courses', 'jobs', or 'gaps'
    
    # Get chats from Firebase
    chats = get_user_chats()
    
    if not chat_id or not question or chat_id not in chats:
        return jsonify({"error": "Invalid request"}), 400
    
    # Get the relevant analysis data
    chat_data = chats[chat_id]
    analysis = chat_data.get('analysis', {})
    
    if analysis_type == 'courses':
        context_data = analysis.get('recommended_courses', [])
        context = json.dumps(context_data, indent=2)
    elif analysis_type == 'jobs':
        context_data = analysis.get('job_descriptions', [])
        context = json.dumps(context_data, indent=2)
    elif analysis_type == 'gaps':
        context_data = analysis.get('skill_gaps', {})
        context = json.dumps(context_data, indent=2)
    else:
        # Use all analysis as context
        context = json.dumps(analysis, indent=2)
    
    # Use Gemini to answer the question based on analysis context
    prompt = f"""
    You are an AI assistant specialized in career development and skills analysis.
    The following is the analysis data for a resume: 
    
    {context}
    
    User question: {question}
    
    Please answer the question thoroughly based on the analysis data provided. 
    If the question cannot be answered with the given information, explain why.
    """
    
    try:
        response = model.generate_content(prompt)
        answer = response.text
        
        # Store this Q&A in the analysis history
        if 'qa_history' not in analysis:
            analysis['qa_history'] = []
            
        analysis['qa_history'].append({
            'question': question,
            'answer': answer,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Update the chat data in Firebase
        chat_data['analysis'] = analysis
        save_chat(chat_id, chat_data)
        
        return jsonify({
            "success": True,
            "answer": answer
        })
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "Failed to generate response"}), 500

@app.route('/analysis_qa_history/<chat_id>', methods=['GET'])
def get_analysis_qa_history(chat_id):
    # Get chats from Firebase
    chats = get_user_chats()
    
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    # Get analysis Q&A history
    analysis = chats[chat_id].get('analysis', {})
    qa_history = analysis.get('qa_history', [])
    
    return jsonify({
        "success": True,
        "qa_history": qa_history
    })

@app.route('/chat/<chat_id>')
def chat(chat_id):
    # Get chats from Firebase
    chats = get_user_chats()
    
    if chat_id not in chats:
        return redirect(url_for('index'))
    
    # Update last accessed timestamp
    chats[chat_id]['last_accessed'] = datetime.now().timestamp()
    save_chat(chat_id, chats[chat_id])
    
    return render_template(
        'chat.html', 
        chat_id=chat_id, 
        chat=chats[chat_id],
        chats=chats
    )

@app.route('/resume_analysis/<chat_id>')
def resume_analysis(chat_id):
    """View detailed resume analysis"""
    # Get chats from Firebase
    chats = get_user_chats()
    
    if chat_id not in chats:
        return redirect(url_for('index'))
    
    chat_data = chats[chat_id]
    
    # Get analysis data
    parsed_resume = chat_data.get('parsed_resume')
    analysis = chat_data.get('analysis', {})
    
    job_descriptions = analysis.get('job_descriptions', [])
    print('Refresh page',job_descriptions)
    skill_gaps = analysis.get('skill_gaps', {})
    recommended_courses = analysis.get('recommended_courses', [])
    qa_history = analysis.get('qa_history', [])
    
    # If we don't have parsed resume data, try getting it from Qdrant
    if not parsed_resume:
        parsed_resume = get_parsed_resume_data(chat_data['resume_id'])
        if parsed_resume:
            chat_data['parsed_resume'] = parsed_resume
            save_chat(chat_id, chat_data)
    
    # If we still don't have analysis data, generate it now
    if not job_descriptions or not skill_gaps or not recommended_courses:
        if parsed_resume:
            job_descriptions = generate_job_descriptions(parsed_resume)
            skill_gaps = analyze_skill_gaps(parsed_resume, job_descriptions)
            recommended_courses = recommend_courses(skill_gaps)
            
            # Update chat data with analysis results
            if 'analysis' not in chat_data:
                chat_data['analysis'] = {}
                
            chat_data['analysis']['job_descriptions'] = job_descriptions
            chat_data['analysis']['skill_gaps'] = skill_gaps
            chat_data['analysis']['recommended_courses'] = recommended_courses
            chat_data['analysis']['last_analyzed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_chat(chat_id, chat_data)
    print(skill_gaps)
    return render_template(
        'resume_analysis.html',
        chat_id=chat_id,
        chat=chat_data,
        parsed_resume=parsed_resume,
        job_descriptions=job_descriptions,
        skill_gaps=skill_gaps,
        recommended_courses=recommended_courses,
        qa_history=qa_history,
        chats=chats
    )

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    chat_id = data.get('chat_id')
    message = data.get('message')
    
    # Get chats from Firebase
    chats = get_user_chats()
    
    if not chat_id or not message or chat_id not in chats:
        return jsonify({"error": "Invalid request"}), 400
    
    # Get resume_id for this chat
    resume_id = chats[chat_id]['resume_id']
    
    # Search for relevant context from the resume
    context = search_resume(message, resume_id)
    
    # Get response from Gemini
    response = get_gemini_response(message, context, resume_id, chat_id)
    
    # Ensure messages list exists
    if 'messages' not in chats[chat_id]:
        chats[chat_id]['messages'] = []
    
    # Add new messages
    chats[chat_id]['messages'].append({
        'role': 'user',
        'content': message,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    chats[chat_id]['messages'].append({
        'role': 'assistant',
        'content': response,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Update last accessed timestamp
    chats[chat_id]['last_accessed'] = datetime.now().timestamp()
    
    # Save updated chat to Firebase
    if save_chat(chat_id, chats[chat_id]):
        return jsonify({
            "response": response
        })
    else:
        return jsonify({"error": "Failed to save message"}), 500

@app.route('/new_chat')
def new_chat():
    return redirect(url_for('index'))

@app.route('/delete_chat/<chat_id>', methods=['POST'])
def delete_chat(chat_id):
    # Delete chat from Firebase
    if delete_chat_from_firebase(chat_id):
        return redirect(url_for('index'))
    else:
        return jsonify({"error": "Failed to delete chat"}), 500

@app.route('/rename_chat/<chat_id>', methods=['POST'])
def rename_chat(chat_id):
    data = request.json
    new_name = data.get('new_name')
    
    # Get chats from Firebase
    chats = get_user_chats()
    
    if chat_id in chats and new_name:
        chats[chat_id]['name'] = new_name
        if save_chat(chat_id, chats[chat_id]):
            return jsonify({"success": True})
    
    return jsonify({"error": "Failed to rename chat"}), 400

@app.route('/refresh_analysis/<chat_id>', methods=['POST'])
def refresh_analysis(chat_id):
    """Refresh the resume analysis (job descriptions, gaps, courses)"""
    # Get chats from Firebase
    chats = get_user_chats()
    
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    chat_data = chats[chat_id]
    resume_id = chat_data.get('resume_id')
    
    # Get parsed resume data
    parsed_resume = chat_data.get('parsed_resume')
    if not parsed_resume:
        parsed_resume = get_parsed_resume_data(resume_id)
    
    # If we still don't have parsed data, we can't proceed
    if not parsed_resume:
        return jsonify({"error": "Resume data not found"}), 404
    
    # Generate fresh analysis using YOUR ORIGINAL LOGIC
    job_descriptions = generate_job_descriptions(parsed_resume)
    skill_gaps = analyze_skill_gaps(parsed_resume, job_descriptions)
    recommended_courses = recommend_courses(skill_gaps)
    
    # Update chat data
    if 'analysis' not in chat_data:
        chat_data['analysis'] = {}
        
    chat_data['analysis']['job_descriptions'] = job_descriptions
    chat_data['analysis']['skill_gaps'] = skill_gaps
    chat_data['analysis']['recommended_courses'] = recommended_courses
    chat_data['analysis']['last_analyzed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Preserve Q&A history if it exists
    if 'qa_history' in chat_data.get('analysis', {}):
        chat_data['analysis']['qa_history'] = chat_data['analysis']['qa_history']
    
    # Save updated chat to Firebase
    if save_chat(chat_id, chat_data):
        return jsonify({
            "success": True,
            "message": "Analysis refreshed successfully"
        })
    else:
        return jsonify({"error": "Failed to save analysis"}), 500

@app.route('/api/courses', methods=['GET'])
def list_courses():
    """API endpoint to list all available courses"""
    # Get all courses from Qdrant
    search_result = qdrant_client.scroll(
        collection_name=course_collection,
        limit=100  # Adjust limit as needed
    )
    
    courses = []
    if search_result[0]:
        for point in search_result[0]:
            courses.append(point.payload)
    
    return jsonify(courses)

@app.route('/api/recommendation/<chat_id>', methods=['POST'])
def get_course_recommendation(chat_id):
    """API endpoint to get course recommendations based on specific skill input"""
    data = request.json
    skills = data.get('skills', [])
    
    if not skills:
        return jsonify({"error": "No skills provided"}), 400
    
    # Get chats from Firebase
    chats = get_user_chats()
    
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    # Get skill gaps data
    skill_gaps = {"Missing technical skills": skills}
    
    # Get recommendations
    recommended_courses = recommend_courses(skill_gaps)
    
    return jsonify({
        "success": True,
        "recommendations": recommended_courses
    })

if __name__ == '__main__':
    print("🚀 Starting FIXED NLP Resume Analyzer with your original logic...")
    print("✅ Fixed Qdrant indexing error")
    print("✅ Preserved your XGBoost ML pipeline for job recommendations")
    print("✅ Fixed course recommendations to use actual missing skills")
    print("✅ Added analysis navigation buttons")
    app.run(debug=True)
