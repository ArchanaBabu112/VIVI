import openai
import os
from dotenv import load_dotenv
import pdfplumber

load_dotenv()

# Set your OpenAI GPT-3 API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def generate_resume_information(pdf_path):
    # Extract text from the provided PDF file
    extracted_text = extract_text_from_pdf(pdf_path)

    # Define the system message and user prompt
    system_message = "You are a helpful assistant trained to extract information from text resumes."
    user_prompt = f"Extract the following information from the provided resume:\n\n" \
                  f"Name:\nContact Details:\nEducation:\nWork Experience:\nSkills:\n\n" \
                  f"Resume Text: {extracted_text}"

    # Make the API request
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Get the generated output
    output = response['choices'][0]['message']['content']

    # Print or use the output as needed
    print(output)

# Example usage:
pdf_file_path = r"C:\Users\ammar\Documents\resume\VishnuPrakash_Resume.pdf"
# generate_resume_information(pdf_file_path)