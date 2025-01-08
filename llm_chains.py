from prompt_templates import prompt_template
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv("HUGGINGFACEHUB_API")

with open("config.yaml","r") as f:
    config=yaml.safe_load(f)

def create_llm(model_path=config["model_path"]["chat_model_llama"],model_type=config["model_type"],model_config=config["model_config"]):
    llm=CTransformers(model=model_path,model_type=model_type,config=model_config)
    return llm

def create_prompt_from_template(template):
    return PromptTemplate(input_variables=["prompt"],template=template)

def load_normal_chain():
    return chatChain()

class chatChain:

    def __init__(self):
        llm=create_llm()
        chat_prompt=create_prompt_from_template(prompt_template)
        # self.llm_chain=LLMChain(llm=llm,prompt=chat_prompt,memory=ConversationBufferWindowMemory(k=1))
        self.llm_chain=LLMChain(llm=llm,prompt=chat_prompt)

    def run(self,user_input):
        return self.llm_chain.invoke(user_input)