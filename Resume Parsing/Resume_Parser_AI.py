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
GOOGLE_API_KEY = "AIzaSyACT4ZQdl9cBC4LMAAHNy79SevZ34fx6jQ"
genai.configure(api_key=GOOGLE_API_KEY)

# Define Resume Parsing Prompt
def parse_resume(text):
    model = genai.GenerativeModel('gemini-1.5-flash')
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
6. Skills
7. Certifications
8. Projects

Additionally, generate a consolidated paragraph summarizing the candidate's background, skills, and experience. This paragraph should be used for similarity measurement between the resume and job description.

Provide the output in structured JSON format.

Resume Text:
{text}
"""

    response = model.generate_content(prompt)
    return response.text

pdf_text = extract_text_from_pdf(r"Sample Resumes\1901841_RESUME.pdf")
parsed_resume = parse_resume(pdf_text)
print(parsed_resume)
