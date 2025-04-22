

import streamlit as st
import json

from PromptGenerator import PromptAgent
from TemplateGenerator import TemplateAgent
from AudioGen import AudioAgent
import warnings
warnings.filterwarnings("ignore")



@st.cache_resource
def load_prompt_agent():
    return PromptAgent()

@st.cache_resource
def load_template_agent():
    return TemplateAgent()

@st.cache_resource
def load_audio_agent():
    return AudioAgent('small')

st.set_page_config(page_title="Reality Architects", layout="wide")

left_col, right_col = st.columns([3, 1])


if "phase" not in st.session_state:

    st.session_state["phase"] = "init"

if "questions" not in st.session_state:
    st.session_state["questions"] = None
if "answer" not in st.session_state:
    st.session_state["answer"] = None
if "feedback" not in st.session_state:
    st.session_state["feedback"] = None
if "emotion" not in st.session_state:
    st.session_state["emotion"] = None
if "prompt" not in st.session_state:
    st.session_state["prompt"] = None
if "finish" not in st.session_state:
    st.session_state["finish"] = False


template_agent = load_template_agent()
prompt_agent = load_prompt_agent()
audio_agent = load_audio_agent()


def generate_av_from_emotion(emotion_text):

    raw_prompt = prompt_agent.ask(
        emotion_text,
        "data/data_prompt/audio",
        "data/data_prompt/visual"
    )

    prompt_json = json.loads(raw_prompt)
    st.session_state["prompt"] = prompt_json

    audio_agent.get(prompt_json["prompt_audio"])

    # Save visual prompt to a file
    # with open("C:\\TouchDesigner projects\\visual_prompt.txt", "w", encoding="utf-8") as f:
    #     f.write(prompt_json["prompt_visual"])
    with open("D:\\TouchDesigner projects\\visual_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt_json["prompt_visual"])

    st.write(f"Generated new audio + visual based on emotion: {emotion_text}")


with right_col:
\
    if st.button("Reset", type="primary"):
        for key in ["phase", "questions", "answer", "feedback", "emotion", "prompt", "finish"]:
            st.session_state[key] = None
        prompt_agent.reset()
        template_agent.reset()
        st.session_state["phase"] = "init" 
        st.rerun()

with left_col:

    if template_agent:

        if st.session_state["phase"] == "init":

            if st.session_state["questions"] is None:
                st.session_state["questions"] = template_agent.query()
                st.write(st.session_state["questions"])

            if st.session_state["answer"] is None:
                user_answer = st.chat_input("Please answer the questions above...")
                if user_answer:
                    st.session_state["answer"] = user_answer

            if st.session_state["answer"] is not None and not st.session_state["finish"]:

                st.session_state["emotion"] = template_agent.query_emotion(st.session_state["answer"])
                st.write(f"Detected Emotion (initial): {st.session_state['emotion']}")


                generate_av_from_emotion(st.session_state["emotion"])
                

                st.session_state["finish"] = True
                st.session_state["phase"] = "feedback"  


        if st.session_state["phase"] == "feedback":
            st.write("**We won't ask the original questions again.**")
            st.write("If you'd like to improve or change the result, please give feedback below.")


            feedback_input = st.chat_input("Feedback? (Leave empty to do nothing)")
            if feedback_input:
                st.session_state["feedback"] = feedback_input

                st.session_state["emotion"] = template_agent.query_emotion(st.session_state["feedback"])
                st.write(f"Detected Emotion (feedback): {st.session_state['emotion']}")


                generate_av_from_emotion(st.session_state["emotion"])


                st.session_state["feedback"] = None  
            else:
                st.write("No new feedback given yet.")
