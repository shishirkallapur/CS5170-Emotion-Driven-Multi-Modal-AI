import os, shutil

import streamlit as st

# from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb.api
from langchain.memory import ConversationBufferMemory

from DocLoaders import DocLoaders


os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'key'
os.environ["DEEPSEEK_API_KEY"] = 'sk-key'
os.environ['OPENAI_API_KEY'] = 'sk-proj-key'


class TemplateAgent:
    def __init__(self, ):

        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.embeddings = OpenAIEmbeddings()
        self.memory_buffer = ConversationBufferMemory(return_messages=True)
        self.llm = ChatOpenAI(model = 'o1-mini')
        self.loaders = DocLoaders()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
        self.documents = None

    def build_prompt(self, context):
        memory_messages = self.memory_buffer.chat_memory.messages
        memory_text = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in memory_messages])

        template = ""
        if memory_text:
            template += "Chat Memory:\n{memory}\n\n"
        if context == "init":
            template += """You are an AI psychological assessor. You have access to the following research {context}. On invocation, do the following:

                            1. Retrieve relevant psychology articles/papers on emotional‑state assessment and question design from the context.
                            2. Formulate three open‑ended, empathetic questions aimed at uncovering the user’s current emotions and intent.
                            3. The questions should not feel repeated and should be somewhat unique.
                            
                            Keep the retrieved information to yourself, and dont print out the papers you got. Just use it to inform yourself.
                        """
        else:
            template += """{question} \n Given this response and the research {context}, analyze them using established psychological frameworks (e.g., appraisal theory, affective circumplex) and the given research.
                            Summarize and show the user’s emotional state in 2 sentences and include a 1–2 sentence rationale grounded in the user’s answers.
                            Create an empathetic response for the user."""
        # print(template)

        prompt = ChatPromptTemplate.from_template(template)
        return prompt.partial(memory=memory_text)
    
    def loader(self, file_path ):
        docs = []
        for file_name in os.listdir(file_path):
            path = os.path.join(file_path, file_name)
            docs.extend(self.loaders.run(path))
        return docs

    def create_retriever(self, key, source):
        
        split_docs = self.text_splitter.split_documents(source)
        vector = Chroma.from_documents(split_docs, self.embeddings)
        
        if key == "context":
            return vector.as_retriever(search_kwargs={"k": 10})
        elif key == "memory":
            return vector.as_retriever(search_kwargs={"k": self.mem_topk})
        else:
            return vector.as_retriever(search_kwargs={"k": self.def_topk})

    def query(self,):
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        prompt = self.build_prompt("init")
        self.documents = self.loader("data/data_template/")
        retriever = {}
        retriever["context"] = self.create_retriever("context", self.documents)
        retriever["question"] = RunnablePassthrough()
        rag_chain = (
            retriever
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        
        response = rag_chain.invoke("")
        self.memory_buffer.chat_memory.add_user_message("")
        self.memory_buffer.chat_memory.add_ai_message(response)
        return response

    
    def query_emotion(self, answer: str):
        prompt = self.build_prompt("answer")
        # documents = self.loader("data/data_template/")
        retriever = {}
        retriever["context"] = self.create_retriever("context", self.documents)
        retriever["question"] = RunnablePassthrough()
        rag_chain = (
            retriever
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        
        response = rag_chain.invoke(answer)
        self.memory_buffer.chat_memory.add_user_message(answer)
        self.memory_buffer.chat_memory.add_ai_message(response)
        return response
    
    def reset(self,):
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.memory_buffer.clear()