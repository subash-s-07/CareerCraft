import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()
import google.generativeai as genai
import os

# Set API Key
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# Define Resume Parsing Prompt
def parse_resume(text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Extract the following details from the resume text:
    - Name
    - Contact Information (Email, Phone)
    - Summary
    - Education (Degree, University, Year)
    - Work Experience (Company, Role, Years)
    - Skills
    - Certifications
    - Projects

    Provide the output in structured JSON format.

    Resume Text: 
    {text}
    """
    response = model.generate_content(prompt)
    return response.text

pdf_text = extract_text_from_pdf(r"Sample Resumes\1901841_RESUME.pdf")
parsed_resume = parse_resume(pdf_text)
print(parsed_resume)
