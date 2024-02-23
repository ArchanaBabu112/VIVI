from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

# Import in the local file of bot_score
from bot_score import Score
# Import tts for text to voice
import tts as tt

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Creating a list to save questions and answers in the database
qa_pairs_list = []


chat = ChatOpenAI()

job_description = """the job should consist of machine learning question we are looking for a candidate with the basics knowledge on machine learning"""


messages_history = [
    SystemMessage(content=f"""you are a job interviewer and your job is to interview candidates according to the company's need
                  
                  Company's job description: {job_description}
                  
                  things that you should ask only after the human response which we will provide shortly(Note you should not ask all at once only ask them one by one after human's response):
                  (
                    welcome the candidate,
                    Ask some common interview questions,
                    Ask job-related questions,
                    Ask questions related to answers,
                    score the answers according to each of his/her reply,
                    you should always ask one question at a time and wait for his/her reply and ask again,
                    say the conclusion and finish the interview it should consist of a string conclusion
                   )
                  """), 
    
    SystemMessage(content="""
                      Rules that should not be broken while having a conversation:
                      (
                          don't ask questions repeatedly,
                          no questions for other jobs,
                          ask questions one by one only after HumanMessage,
                          should not give the job if the person is speaking ill-mannered, 
                          should give the candidate a score based on these rules and also his performance at the end,
                          )
                          
                        These rules are strict and should not be broken
                        """)
]



## Resume score


import resume_extractor

re = resume_extractor
pdf_path =  r"C:\Users\ammar\Documents\resume\VishnuPrakash_Resume.pdf"
text_resume = re.extract_text_from_pdf(pdf_path)
# print(text_resume)

# Initializing bot_score object as sc
sc = Score()

resume_score = sc.resume_scoring(text_resume,job_description)
# print("The overall score of this resume is :",resume_score)






# Importing a function from tts in order to select the voice and model version for speech
selected_voice, selected_model = tt.select_voice_model()

# Importing audio_transcriptor
import audio_transcription as a_t

# Limit for asking questions or stopping the for loop
no_of_times_to_ask_question = 20

for i in range(no_of_times_to_ask_question):
    ai = chat(messages=messages_history).content
    
    
    ret = tt.ask_question_and_repeat(selected_voice, selected_model, ai)
    
    # print(ai)
    if ret == True:
        response = a_t.record_and_transcribe()
        chats = response
    
    each_score = sc.answer_score(ai, chats, job_description)
    
    # Taking history of question, answer, score in a list called qa_pairs_list
    qa_pairs_list.append({
        "question": ai,
        "answer": chats,
        "score": each_score
    })
    
    # print(qa_pairs_list)
 
 ## No need  
    ## messages_history.append(AIMessage(ai))
    ## messages_history.append(HumanMessage(chats))
    
    
# scoring_list = []
# scoring_list.append(
    
# )
