

import os
import dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class Score:
    
    total_score = 0
    resume_score = 0
    overall_total = 0
    
    score_llm = OpenAI(temperature=.4)
    resume_llm = OpenAI(temperature=.5)
    
        
    def answer_score(self, question, answer, job_description):
        """parameters:
        
        question: question that is asked in the interview
        answer: answer that the candidate has given
        job_description: job description for hiring
        
        Returns: the score for the answer
        """
        
        # llm model for calculating the score 
        llm = self.score_llm

        
        template_2 =  """
        you are an scoring partner for interview questions for a company:
        
        there are two scores that should be calculated for the final score:
        answer_score and importance_score
        
        scoring of these two scores should not be influenced by each other 
    
        
        
        calculate the answer_score based on the following:
        
        
        you should score this out of 10:
        scores should be between 1 to 10
        
        if the you dont know the answer or cant find the answer then give 7 or above as the score else:
        
        7 or above 7 if the candidate answered correctly
        5 or above 5 if the candidate answered but need to improve 
        3 or above 3 if the  candidate have some knowledge about it
        1 or above 1 if the answer is incorrect
        

        
        score based on: 
        * honesty
        * accuracy of answers 
        * manners
        
        should reduce the points:
        * unappropriate behaviour 
        * dont know the answers
        * not giving proper answer
        * not speaking proper english
        
        
        calculate the importance_score based on the following:
        
        * importance of question in the job_description for hiring the candidate
        * importance_score should be between 1 and 2
        * 1 for not much important question for hiring
        * 2 for important question for hiring

        
        details regarding the question, answer and job description are given below
            
        Company Job description: {job_description}
        
        Question: {question}

        Answer: {answer}
        
        Company Job description is the criteria given by the company for hiring candidates
        
        Question is the question asked by the interviewer
        
        Answer is the answer given by the candidate
        
        Only return the score no need of any description should only answer in digits and nothing else
        Only in this format and nothing else:
        
        
        
        Importance_Score: ..
        Answer_Score: ..
        
        """
        
        
        
        # template for scoring
        prompt = PromptTemplate(
            input_variables=['question', 'answer', 'job_description'],
            template=template_2
        )
        
        # chaining the model and the template  
        chain = LLMChain(llm=llm, prompt=prompt, output_key='score')
        
        # response from the llm model 
        response = chain.invoke({'question': question, 'job_description': job_description, 'answer': answer})
        
        # cleaning the score output from llm
        scores = response['score'].strip().split()
        
        # cleaned scores
        scores = list(filter(lambda x: x.isdigit(), scores))
        
        # importance of scores and the answer score 
        score_weight, score = list(map(lambda x: int(x), scores))
        
        # final score of the answer 
        final_score = score_weight * score
        
        # updating the overall total score and total score gained to get the final result after the interviews  
        self.overall_total += score_weight * 10
        self.total_score += final_score
        
        # returning the current score for the answer
        return final_score 


    def resume_scoring(self, resume, job_description):
            
        llm = self.resume_llm
        
        
        template_1 = """
        
        you are an evaluation program for resume:
        
        the score would be based on the following details:
        * resume neatness
        * skills that have direct influence on job description
        * the clarity of the descriptions that is provided in resume
        * the projects that are mentioned which is related to the job description
        * if the resume is really good score really high and if it is really bad score really low be accurate
        
        score should be between 1 to 4: 
        4 means really good resume 
        3 means resume is good and average
        2 means resume can be considered but can be better
        1 means resume is bad
        
        the details of the resume and Job Description are given below:
        Resume: {resume}
        Job Description: {job_description}
        
        Only return the score no need of any description should only answer in digits and nothing else
        Only in this format and nothing else:
        
        Score: ..
        
        """
        
        
        # template for scoring
        prompt = PromptTemplate(
            input_variables=['resume', 'job_description'],
            template=template_1
        )
        
        # chaining the model and the template 
        chain = LLMChain(llm=llm, prompt=prompt, output_key='score')
        
        # response from the llm model 
        response = chain.invoke({'job_description': job_description, 'resume': resume})
        
        # cleaning the score output from llm
        score = response['score'].strip().split()
        
        # getting the cleaned scores 
        score = list(filter(lambda x: x.isdigit(), score))
        resume_score = int(score[0])
        
        self.resume_score = resume_score
        
        return resume_score
    
    
    def get_total_score(self):
        answer_score, resume_score = self.get_all_scores()
        return answer_score + resume_score
        
    def get_all_scores(self):
        """
        Returns:
           answer_score and resume_score
        """
        answer_score = (self.total_score / self.overall_total) * 17
        resume_score = self.resume_score
        return answer_score, resume_score
    
    
if __name__ == "__main__":
        
    question = "what is your dads name"
    answer = "my dads name is afsal and he is a farmer"

    # try out details:
    question1 = "what is machine learning"
    answer1 = 'machine learning is a technique used for predicting data from the data that it has trained. it is used in object detection, stock prediction and etc'
    job_description = 'we are looking for a machine learning fresher who has indepth knowledge in python'

    resume = """
    Afsal
    123 Main Street, City, State, Zip Code
    Phone: (555) 123-4567 | Email: johndoe@email.com
    LinkedIn: linkedin.com/in/johndoe

    Objective:
    Enthusiastic and detail-oriented Machine Learning Engineer with a strong foundation in data analysis and predictive modeling. Seeking to leverage advanced skills in machine learning algorithms and programming languages to contribute effectively to cutting-edge projects in a dynamic team environment.

    Education:
    Bachelor of Science in Computer Science
    University of Technology, City, State
    Graduated: May 2020

    Skills:
    - Proficient in Python, R, and Java
    - Experience with machine learning frameworks such as TensorFlow and PyTorch
    - Strong understanding of algorithms and data structures
    - Data preprocessing and feature engineering
    - Statistical analysis and hypothesis testing
    - Deep learning techniques including CNNs and RNNs
    - Experience with version control systems (Git)
    - Excellent problem-solving and analytical skills
    - Strong communication and teamwork abilities

    Experience:
    
    Machine Learning Intern
    XYZ Tech Company, City, State
    June 2020 - August 2021
    - Developed and implemented machine learning models for predictive maintenance, resulting in a 15% reduction in downtime.
    - Conducted data analysis and preprocessing on large datasets to extract meaningful insights.
    - Collaborated with cross-functional teams to integrate machine learning solutions into existing products.
    - Contributed to the development of a recommendation system, improving user engagement by 20%.

    Software Developer Intern
    ABC Solutions, City, State
    May 2019 - August 2019
    - Assisted in the development of a web-based application using Java Spring framework.
    - Implemented RESTful APIs to enable seamless communication between front-end and back-end systems.
    - Conducted unit testing and debugging to ensure the quality and reliability of the software.

    Projects:

    Sentiment Analysis of Customer Reviews
    - Developed a sentiment analysis model using natural language processing techniques to analyze customer reviews.
    - Achieved an accuracy rate of 85% on sentiment classification tasks.

    Image Classification using Convolutional Neural Networks
    - Implemented a CNN model in TensorFlow to classify images from the CIFAR-10 dataset.
    - Achieved a classification accuracy of 90% on test data.

    Certifications:
    - Machine Learning Certification, Coursera
    - Deep Learning Specialization, Udacity

    Languages:
    English (Native), Spanish (Proficient)

    """

    score = Score()

    total = 0

    for i in range(3):
        print(score.answer_score(question, answer, job_description))
        print(score.answer_score(question1, answer1, job_description))
        print(score.resume_scoring(resume=resume, job_description=job_description))
        n=score.get_total_score()
        print(n)
        total += n


    # take the average score from three evaluation of the same question 
    print(total/3)
                                    