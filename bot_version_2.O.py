from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, SystemMessage, HumanMessage
import audio_transcription as a_t

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



import tts




from bot_score import Score




class Assistance:
    
    def __init__(self, job_description):
        self.chat = ChatOpenAI(temperature=1)
        self.job_description = job_description
        self.messages = [
            SystemMessage(content=f"""you are a job interviewer and your job is to interview candidate according to company's need
                        
                        At the start of the interview welcome the candidate once and only once
                        
                        Company's job description: {job_description}
                        
                        questions that should be asked in the interview:
                        (
                            * should ask at least 12 questions
                            * welcome the candidate at the start of the interview
                            * ask questions about personal information
                            * ask questions about how to handle certain cituations in the company
                            * ask about 6 or more questions according to the job description,
                            * you should always ask one question at a time and wait for his/her reply and ask again,
                            * say the conclusion and finish the interview and at the end say "the interview is finished you can exit now",

                        )
                        
                        You should only ask questions and nothing else
                        
                        """), 
            
                SystemMessage(content="""
                            Rules that should not be broken while having a conversation:
                            (
                                dont ask questions repeatedly,
                                no questions for other jobs,
                                ask questions one by one only after the candidate messages,
                                should not give any information at the end whether candidate has passed or failed the interview
                                )
                                
                                These rules are strict and should not be broken
                                """)
        ]

    def get_question(self):
        ai_question = self.chat.invoke(input=self.messages).content
        self.messages.append(AIMessage(ai_question))
        return ai_question

    def answer_question(self, answer):
        self.messages.append(HumanMessage(answer))



class Q_and_A():
    
    def __init__(self,job_description):
        self.assist = Assistance(job_description)
        self.jb = job_description
        self.curent_question=''
        self.current_answer = ''
        self.each_scores = []


        
    def ask_quesiton(self, repeat):
        
        selected_voice,selected_model = "nova","tts-1-hd"
        
        if repeat == 1:
            ai_question = self.curent_question
        else:
            ai_question = self.assist.get_question()
            self.curent_question = ai_question
             
        
        
       
        ret = tts.ask_question_and_play(selected_voice, selected_model, ai_question)
        return ret,ai_question
        
    def speak_answer(self):
        
        answer = a_t.record_and_transcribe()
        
        self.assist.answer_question(answer=answer)
        
        return answer
    
    def question_answer_score(self):
        ret,ai_question = self.ask_quesiton(0)
        answer = self.speak_answer()
        score = Score()
        each_socre = score.answer_score(ai_question,answer,self.jb)  
        self.each_scores.append({
            'question':ai_question,
            "answer":answer,
            "q_a_score": each_socre
        }
        )
        
        return self.each_score
    
    
qa = Q_and_A("This is for a machine learning interview")
qa.ask_quesiton(0)
qa.speak_answer()
qa.ask_quesiton(0)


        
        
        
        

        
        