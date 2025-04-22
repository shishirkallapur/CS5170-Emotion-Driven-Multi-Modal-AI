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
os.environ["DEEPSEEK_API_KEY"] = 'key'
os.environ['OPENAI_API_KEY'] = 'sk-proj-key'


class PromptAgent:
    
    def __init__(self,):
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=3000)
        self.embeddings = OpenAIEmbeddings()
        self.mem_topk = 3
        self.doc_topk = 14
        self.def_topk = 5
        
        self.memory_buffer = ConversationBufferMemory(return_messages=True)
        self.response_counter = 0
        self.llm = ChatOpenAI(model = 'o1-mini')
            
        self.loaders = DocLoaders()





    def loader(self, file_path):
        docs = []
        if os.path.isfile(file_path): # MIGHT BE A POTENTIAL ISSUE BUT NOT USED SO FAR SO HAVE TO FIX LATER
            docs.extend(self.loaders.run(file_path))
            
        elif os.path.isdir(file_path):
            for file_name in os.listdir(file_path):
                path = os.path.join(file_path, file_name)
                docs.extend(self.loaders.run(path))
        else:
            print(f"'{file_path}' does not exist or is a special file type")
            
        return docs
    
    def build_prompt(self, ):
        memory_messages = self.memory_buffer.chat_memory.messages
        memory_text = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in memory_messages])

        template = ""
        if memory_text:
            template += "Chat Memory:\n{memory}\n\n"
        # if len(context) != 0:
        #     template += "Context:\n{context}\n\n"
        # """You are a creative prompt generator. Input: the user’s emotional state {question}. Perform the following:

        #                 1. Retrieve articles/papers on music–emotion relationships . {context_audio}.
        #                 2. Generate a **Music‑Generation Prompt** that:
        #                     - Specifies the target emotion.
        #                     - Indicates whether the music should align with or counter the emotion.
        #                     - Suggests mood descriptors (e.g., “warm,” “tense”), tempo, instrumentation, and any structural notes (e.g., “build to a climax”).
        #                 3. Retrieve articles/papers on visual‑emotion mapping. {context_visual}.
        #                 4. Generate a **Visual‑Generation Prompt** that:
        #                     - Describes the imagery, color palette, and artistic style to evoke or counter the emotion.
        #                     - Includes atmosphere/mood keywords (e.g., “ethereal,” “surreal,” “moody shadows”).
        #                 Give the response in a dictionary with dictionary keys as prompt_audio and prompt_visual. 
        #                 DO NOT include json tags \n Example response: {{\"prompt_audio\": str, \"prompt_visual\": str}}"""

        template += """You are an emotional analysis and creative generation assistant. A user provides an emotional update {question}. Your task is to:

                    1. Identify the primary **emotion** (e.g., anger, sadness, joy, anxiety, fear, love).
                    2. Determine the **intent** of the user regarding this emotion (e.g., express it, calm down, understand it, uplift mood, sustain joy, ground themselves, explore the emotion).
                    3. Generate an appropriate **audio generation prompt** and a **visual generation prompt** based on both the emotion and intent. Use articles/papers on music–emotion relationships: {context_audio} and on visual‑emotion mapping: {context_visual}.
                    - Suggest mood descriptors (e.g., “warm,” “tense”), tempo, instrumentation, and any structural notes (e.g., “build to a climax”).
                    - Describe the imagery, color palette, and artistic style to evoke or counter the emotion.
                    - Include atmosphere/mood keywords (e.g., “ethereal,” “surreal,” “moody shadows”).
                    Be nuanced and avoid making assumptions — extract intent only if implied or clearly stated. Use the emotional tone and keywords to guide your inference.

                    ### Example User Input:
                    "I'm so frustrated right now. Everything feels out of control and I just want to scream or break something."

                    ### Output:
                    Emotion: Anger  
                    Intent: Express  
                    Audio Prompt: Generate a heavy metal audio with distorted guitars, aggressive drums, and chaotic energy to reflect emotional release.  
                    Visual Prompt: Create a jagged, high-contrast visual using dark red and black tones with glitch effects and explosive movement to express raw frustration.

                    ---
                    Give the response in a dictionary with dictionary keys as prompt_audio and prompt_visual. 
                    DO NOT include json tags \n Example response: {{\"prompt_audio\": str, \"prompt_visual\": str}}
                    """
        
        # print(template)

        prompt = ChatPromptTemplate.from_template(template)
        return prompt.partial(memory=memory_text)
    
    def ask(self, question, context_path_audio, context_path_visual):
        
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        
        # load the docs
        documents_audio = self.loader(context_path_audio)
        # load the docs
        documents_visual = self.loader(context_path_visual)
        
        # load the mem
        # memory = self.loader(mem_path)
        
        # build prompt 
        prompt = self.build_prompt()
        
        # create retrievers
        retriever = {}
        

        retriever["context_audio"] = self.create_retriever("context_audio", documents_audio)
        retriever["context_visual"] = self.create_retriever("context_visual", documents_visual)
        
    
        
        retriever["question"] = RunnablePassthrough()
        rag_chain = (
            retriever
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        
        response = rag_chain.invoke(question)
        self.memory_buffer.chat_memory.add_user_message(question)
        self.memory_buffer.chat_memory.add_ai_message(response)
        return response
        
    def create_retriever(self, key, source):
        
        split_docs = self.text_splitter.split_documents(source)
        vector = Chroma.from_documents(split_docs, self.embeddings)
        
        if "context" in key:
            return vector.as_retriever(search_kwargs={"k": self.doc_topk})
        elif key == "memory":
            return vector.as_retriever(search_kwargs={"k": self.mem_topk})
        else:
            return vector.as_retriever(search_kwargs={"k": self.def_topk})
        
    def reset(self,):
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.memory_buffer.clear()
        # for filename in os.listdir(context_path):
        #     file_path = os.path.join(context_path, filename)
        #     try:
        #         if os.path.isfile(file_path) or os.path.islink(file_path):
        #             os.unlink(file_path)
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)
        #     except Exception as e:
        #         print('Failed to delete %s. Reason: %s' % (file_path, e))
                
        # for filename in os.listdir(mem_path):
        #     file_path = os.path.join(mem_path, filename)
        #     try:
        #         if os.path.isfile(file_path) or os.path.islink(file_path):
        #             os.unlink(file_path)
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)
        #     except Exception as e:
        #         print('Failed to delete %s. Reason: %s' % (file_path, e))
        