import google.generativeai as genai
import pdfplumber
import pandas as pd
import os
import time
import google.api_core.exceptions

# Configure Gemini API Key
genai.configure(api_key="AIzaSyACT4ZQdl9cBC4LMAAHNy79SevZ34fx6jQ")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF resume."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

def get_gemini_score(resume_text, job_description, model_name="gemini-1.5-flash", max_retries=5):
    """Gets similarity score using Gemini API with retries and error handling."""
    
    prompt = f"""
    Given the following resume and job description, provide a similarity score between 0 and 1.
    A score of 1 means a perfect match, while 0 means no relevance at all.

    Resume:
    {resume_text}  # Limiting input size to avoid exceeding API token limit
    
    Job Description:
    {job_description}

    Provide only the score as a float between 0 and 1.
    """

    model = genai.GenerativeModel(model_name)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            score = float(response.text.strip())
            return max(0, min(1, score))  # Ensures score is in [0,1] range
        except (ValueError, AttributeError):
            print(f"⚠️ Invalid API response on attempt {attempt + 1}. Retrying...")
        except google.api_core.exceptions.ResourceExhausted:
            wait_time = 10 + (5 * attempt)  # Smart delay increasing with attempts
            print(f"🚦 Rate limit exceeded. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    return None

# Load Resume
pdf_path = r'E:\SEM 8\NLP\NLP-Project\Sample Resumes\candidate_018.pdf'
resume_text = extract_text_from_pdf(pdf_path)

# Load Job Descriptions
job_descriptions = pd.read_csv(r'E:\SEM 8\NLP\NLP-Project\jobs.csv')['Job_Des'].tolist()

# Load or Create CSV
data=pd.DataFrame([])
data.to_csv('4.csv')
output_file = '4.csv'
data = pd.read_csv(output_file)
data = pd.DataFrame(columns=['Job_Description', 'Score'])
print(data)

# Process All Job Descriptions Sequentially
for job_desc in job_descriptions:
    if job_desc in data['Job_Description'].values:
        print(f"✅ Skipping already processed job: {job_desc[:50]}...")
        continue  # Skip if already processed

    score = get_gemini_score(resume_text, job_desc)
    print(f"Score: {score} for job: {job_desc}")

    # Append result
    new_entry = pd.DataFrame([{"Job_Description": job_desc, "Score": score}])
    data = pd.concat([data, new_entry], ignore_index=True)

    # Save Progress
    data.to_csv(output_file, index=False)

print("✅ All job descriptions processed successfully! Data saved to 1.csv.")
