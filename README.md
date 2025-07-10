# CareerCraft

CareerCraft is an AI-powered resume analysis and career development platform that helps job seekers optimize their resumes and find suitable job opportunities and skill development courses.

## Overview

CareerCraft uses Natural Language Processing (NLP) and Machine Learning to:

1. **Parse and analyze resumes** - Extract key information from PDF and DOCX resume files
2. **Recommend job opportunities** - Match resume skills to relevant job listings
3. **Identify skill gaps** - Analyze missing skills required for desired roles
4. **Suggest courses** - Recommend relevant Coursera courses to fill skill gaps
5. **Provide an AI assistant** - Answer questions about career development and resume improvement

## Key NLP Applications

- **Resume Information Extraction**: Structured information extraction from unstructured resume text
- **Semantic Skill Matching**: Vector-based skill matching between resumes and job descriptions
- **Domain Classification**: Identifying primary domains/fields from extracted skills
- **Text Summarization**: Creating concise resume summaries for easier matching
- **Context-Aware Question Answering**: AI assistant that answers questions based on resume context
- **Skill Gap Identification**: NLP-based analysis of missing skills through comparative analysis

## Features

- **Resume Parsing**: Upload PDF or DOCX files and extract structured information including name, contact details, education, work experience, and skills
- **Job Recommendations**: Get personalized job recommendations based on your skills and experience
- **Skill Gap Analysis**: Identify missing skills that are in demand for your target roles
- **Course Recommendations**: Discover relevant courses from Coursera to develop missing skills
- **AI Chat Assistant**: Ask questions about your resume, career path, or get advice on skill development
- **Persistent Storage**: All analyses and chat history are saved to Firebase for future reference

## Technology Stack

- **Backend**: Flask (Python)
- **AI & ML**:
  - Google Generative AI (Gemini 1.5 Flash)
  - Sentence Transformers for embeddings
  - XGBoost for job ranking
- **Vector Database**: Qdrant for semantic search
- **Storage**: Firebase Firestore
- **Document Processing**: PyPDF2, docx2txt
- **Web Scraping**: BeautifulSoup4, Requests
- **Frontend**: HTML, CSS, JavaScript

## NLP Techniques & Models

### Transformer Models
- **Sentence Transformers**: `all-MiniLM-L6-v2` model for generating dense vector embeddings of text
- **Google Gemini 1.5 Flash**: Large language model for parsing resumes, generating skill gap analyses, and powering the AI assistant

### Vector Embeddings & Semantic Search
- **Vector Dimension**: The embedding model generates fixed-size vectors (384 dimensions)
- **Similarity Metric**: Cosine similarity for measuring relevance between queries and stored vectors
- **Chunking Strategy**: Text documents are chunked into 300-word segments with 50-word overlaps for better semantic search
- **Retrieval Augmented Generation (RAG)**: Uses vector search to retrieve relevant resume contexts before generating AI responses

### Machine Learning
- **XGBoost Model**: Trained gradient boosting model for ranking job recommendations based on skill relevance
- **Ranking Features**: Uses domain skills, experience level, and job requirements as features
- **Model Serialization**: Saved as `xgb_model.json` for consistent inference

### Text Processing
- **Named Entity Recognition**: Used for identifying skills, education, experience, and other structured information
- **Regex Pattern Matching**: For extracting contact information, dates, and formatting structured data
- **JSON Response Parsing**: For handling API responses and extracting structured data

### Data Storage
- **Vector Indexing**: Resume chunks and course descriptions are stored as vector embeddings in Qdrant
- **Payload Indexing**: Additional metadata like resume_id is indexed for efficient filtering
- **Document Structure**: Hierarchical data storage in Firestore for resume analyses, chat history, and user data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CareerCraft.git
cd CareerCraft
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Firebase:
   - Create a Firebase project at [firebase.google.com](https://firebase.google.com)
   - Generate a service account key and save it as `static/nlp-project-d88b9-firebase-adminsdk-fbsvc-3164e3ecb3.json`

5. Set up Google Generative AI:
   - Get an API key for Google Generative AI (Gemini)
   - Add your API key to the `GOOGLE_API_KEY` variable in `app.py`

6. Set up Qdrant:
   - You can use Qdrant Cloud or run it locally
   - Update the Qdrant client configuration in `app.py`

## Usage

1. Start the application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Upload your resume and wait for the analysis to complete
4. Explore job recommendations, skill gaps, and course suggestions
5. Chat with the AI assistant for personalized career advice

## Project Structure

- `app.py` - Main Flask application
- `course_scrap.py` - Web scraper for Coursera courses
- `Dataset_Generation.py` - Data generation utilities
- `model_ranking.ipynb` - Notebook for training the job ranking model
- `xgb_model.json` - Trained XGBoost model
- `Processing_Files/` - Files for data processing
- `Resume Parsing/` - Resume parsing utilities
- `Sample Resumes/` - Sample resume files for testing
- `skill_extractor_models/` - Models for skill extraction
- `static/` - Static assets and Firebase configuration
- `templates/` - HTML templates for the web interface
- `Web Scrapping/` - Web scraping utilities

## How It Works

1. **Resume Upload and Parsing**:
   - User uploads a resume (PDF/DOCX)
   - System extracts text and uses Gemini AI to parse structured information
   - Resume text is chunked and indexed in Qdrant for semantic search

2. **Job Recommendation**:
   - System identifies domain skills from the resume
   - XGBoost model ranks and returns relevant job listings
   - Jobs are displayed with relevance scores and match details

3. **Skill Gap Analysis**:
   - System compares resume skills with job requirements
   - Gemini AI identifies technical, soft skill, experience, and certification gaps
   - Results are presented in an organized format

4. **Course Recommendation**:
   - System scrapes Coursera for courses related to missing skills
   - Courses are ranked by relevance to the identified skill gaps
   - Course details include titles, descriptions, and skill coverage

5. **AI Assistant**:
   - User can ask questions about their resume, jobs, or skills
   - System uses semantic search to find relevant resume sections
   - Gemini AI generates contextual responses based on resume data

## Implementation Details

### Resume Parsing Pipeline
```
Upload → Text Extraction → Chunking → Gemini Structured Parsing → Vector Embedding → Qdrant Indexing
```

### Job Recommendation Pipeline
```
Resume Skills → Domain Skill Extraction → XGBoost Ranking → Job Matching → Relevance Scoring
```

### Skill Gap Analysis Pipeline
```
Resume Skills + Job Requirements → Semantic Comparison → Gap Identification → Categorization → Presentation
```

### Course Recommendation Pipeline
```
Skill Gaps → Coursera Scraping → Course Embedding → Vector Similarity Search → Relevance Ranking → Presentation
```

### AI Assistant Pipeline
```
User Query → Query Embedding → Semantic Search → Context Retrieval → Gemini Response Generation
```
