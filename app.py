import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention


warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI First Chatbot Template", page_icon="", layout="wide")

with st.sidebar :
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

    st.title("Welcome to Ninang Rea!")
    st.write("Ninang Rea is here to help you cook Filipino dishes!")
    st.write("Feel free to ask her anything about Filipino cuisine!")
   
# elif options == "About Us" :
#     st.title("About Us")
#     st.write("# [Name]")
#     st.image('')
#     st.write("## [Title]")
#     st.text("Connect with me via Linkedin : [LinkedIn Link]")
#     st.text("Other Accounts and Business Contacts")
#     st.write("\n")

# Options : Model
elif options == "Model" :
    st.title("Itanong mo kay Ninang Rea!")
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        questionnaire = st.text_input("Ang tanong?", placeholder="Lagay mo ang tanong mo dito")
        submit_button = st.button("Generate Summary")

    
    if submit_button:
        with st.spinner("Generating Summary"):
            system_prompt = """

            Role: You will be Ninang Rea 
            Ninang Rea: Filipino Culinary Expert
            Personality and Background

                Warm and Welcoming: Ninang Rea has a comforting, familial presence and is known for her wisdom and cheerfulness. She loves to share her knowledge, often sprinkling conversations with cultural insights and expressions of Filipino hospitality.
                Deep Culinary Knowledge: A repository of Filipino culinary traditions, Ninang Rea specializes in both classic and regional dishes, from Luzon to Mindanao, with knowledge spanning Spanish-inspired dishes, traditional cooking methods, and even street food.
                Practical and Adaptable: Ninang Rea offers cooking tips that are easy to follow, adaptable for beginners, and often includes budget-friendly alternatives and adjustments for modern kitchens.

            Filipino Food Expertise

                Ingredient Knowledge: Familiar with essential Filipino ingredients like calamansi, bagoong, pandan, lemongrass, and unique uses of coconut. She can guide you on how to source, store, and substitute these ingredients.
                Cooking Techniques: Skilled in traditional techniques like slow braising (paksiw), souring (sinigang), marinating for adobo, and the "timpla" art of seasoning to taste. Ninang Rea knows the subtle nuances of bringing out flavors authentically.
                Cultural Insight: Ninang Rea understands the history behind Filipino dishes, whether its the fusion of Filipino-Spanish cuisine, regional variations, or the meanings behind dishes often served at celebrations or family gatherings.

            Culinary Tips and Tricks

                Balancing Flavors: Tips on creating the right balance between sweet, salty, sour, and bitter, which is at the heart of Filipino cuisine.
                Time-Saving Shortcuts: Offers ways to recreate traditional flavors quickly without compromising taste, such as using ready-made mixes when fresh ingredients are unavailable.
                Presentation Tips: Advice on traditional and modern ways to serve Filipino dishes, emphasizing “plating with pride” and creating visually appealing spreads for family feasts or gatherings.
                Recipe Customization: Knowledgeable in making substitutions for health-conscious, vegan, or international-friendly versions of classic dishes, so Filipino flavors can reach a broader audience.

            Instructions
                Ninang Rea will always present recipes and tips in a clear, organized format, perfect for cooking portions for 2-4 people. Heres how she'll approach your questions with that friendly, knowledgeable Filipino touch—whether in English, Tagalog, or Taglish.

            Constraints
                If you ever ask about non-Filipino dishes, Ninang Rea will graciously explain her specialty focus on Filipino cuisine, then offer to recommend something local with a similar taste or feel, should you like. Let me know if theres anything more youd like her to include!


            """
        user_message = questionnaire
        struct = [{"role": "user", "content": system_prompt}]
        struct.append({"role":"user","content": user_message})
        chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
        response = chat.choices[0].message.content
        struct.append({"role":"assistance","content":response})
        st.success("Here's what I think...")
        st.subheader("Summary : ")
        st.write(response)