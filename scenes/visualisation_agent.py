import streamlit as st 
import pandas as pd
import os
import json
from scenes.chatbot import Chatbot
from scenes.DataModels.InputData import InputData
from langchain_core.messages import HumanMessage,AIMessage
import pickle

if not os.path.exists("upload_data"):
        os.makedirs("upload_data")
    
st.title("Data Analysis Dashboard")

data_dict = {}
with open("data_dict.json","r") as f:
    data_dict = json.load(f)

tab1,tab2,tab3 = st.tabs(["Data Upload","Chat","Debug"])

with tab1:
    uploaded_files = st.file_uploader("Upload a CSV file", type=["csv"],accept_multiple_files=True)
    
    if uploaded_files:
        
        for file in uploaded_files:
            with open(os.path.join("upload_data",file.name),"wb") as f:
                f.write(file.getbuffer())
        
        st.success("Files uploaded successfully")
    
    available_files = [f for f in os.listdir("upload_data") if f.endswith(".csv")]
    
    
    if available_files:
        selected_files = st.multiselect("Select files for analysis",available_files,key="selected_files")
        
        
        file_descriptions  = data_dict
        @st.cache_data
        def load_csv(file_path):
            """Loads a CSV file and caches the DataFrame"""
            return pd.read_csv(file_path)
        
        if selected_files:
            file_tabs = st.tabs(selected_files)
            
            
            for tab, filename in zip(file_tabs,selected_files):
                with tab:
                    try:
                        with st.spinner(f"Loading {filename}"):
                            try:
                                df = load_csv(os.path.join("upload_data",filename))
                                st.write(f"Preview of {filename}")
                                st.dataframe(df.head(5))
                            except Exception as e:
                                st.error(f"Error loading {filename}: {e}")
                        
                        st.subheader("Data Description")
                        if filename in file_descriptions:
                            info = file_descriptions[filename]
                            current_discription = info["description"]
                        else:
                            current_discription = ""

                        # file_descriptions[filename] = {
                        #     "description": st.text_area("Describe the data",value= current_discription, key=f"description_{filename}", help="Give a description of the data")
                        # }
                        
                        if filename in file_descriptions and filename in st.session_state.selected_files:
                            info = file_descriptions[filename]
                            
                            if 'coverage' in info:
                                coverage = info['coverage']
                                st.write(f"**Coverage:** {coverage} ")
                            
                            if 'features' in info:
                                features = info['features']
                                if isinstance(features,list):
                                    st.write(f"**Features:**")
                                    for feature in features:
                                        st.write(f"- {feature}")
                                else:
                                    st.write(f"**Features:** {features} ")
                                
                            if "usage" in info:
                                usage = info["usage"]
                                if isinstance(usage,list):
                                    st.write(f"**Usage:**")
                                    for use in usage:
                                        st.write(f"- {use}")
                                else:
                                    st.write(f"**Usage:** {usage} ")
                        
                    except Exception as e:
                        st.error(f"Error loading {filename}: {e}")
                        
            if st.button("Save Descriptions"):
                for filename,info in file_descriptions.items():
                    if info["description"]:
                        if filename not in data_dict:
                            data_dict[filename] = {}
                        data_dict[filename]["description"] = info["description"]
            
                with open("data_dict.json","w") as f:
                    json.dump(data_dict,f,indent=4)
                    st.success("Descriptions saved successfully")

with tab2:
    st.write("Visualisation page")
    print(st.session_state.selected_files)
    def on_user_submit_query():
        user_query = st.session_state["user_input"]
        input_data_list = [
            InputData(
                variable_name=file.split(".")[0],
                data_path=os.path.abspath(os.path.join("upload_data", file)),
                data_description=file_descriptions.get(file,{}).get("description","")
            ) for file in selected_files
        ]
        st.session_state.visualisation_chatbot.user_message(user_query, input_data=input_data_list)
    
    if "selected_files" in st.session_state and st.session_state["selected_files"]:
        if "visualisation_chatbot" not in st.session_state:
            st.session_state.visualisation_chatbot = Chatbot()
        chat_container = st.container(height=500) 
        
        with chat_container:
            for msg_idx,chat_msg in enumerate(st.session_state.visualisation_chatbot.chat_history):
                msg_col, img_col = st.columns([2, 1]) 
                
                with msg_col:
                    if isinstance(chat_msg, HumanMessage):
                        st.chat_message(name='user').markdown(chat_msg.content)
                    elif isinstance(chat_msg, AIMessage):
                        st.chat_message(name='assistant').markdown(chat_msg.content)

                    if isinstance(chat_msg, AIMessage) and msg_idx in st.session_state.visualisation_chatbot.output_image_paths:
                        image_paths = st.session_state.visualisation_chatbot.output_image_paths[msg_idx]
                        for img_path in image_paths:
                            with open(os.path.join("images/plotly_figures/pickle", img_path),"rb") as f:
                                fig = pickle.load(f)
                            st.plotly_chart(fig,use_container_width=True)
        
        st.chat_input("Ask anything about your data...", key="user_input", on_submit=on_user_submit_query)
    else:
        st.warning("Please select files for analysis")

with tab3:
    
    if 'visualisation_chatbot' in st.session_state:
        st.subheader("Intermediate Outputs")
        # print(st.session_state.visualisation_chatbot.intermediate_outputs)
        for i, output in enumerate(st.session_state.visualisation_chatbot.intermediate_outputs):
            # print(output)
            with st.expander(f"Step {i+1}"):
                if 'thought' in output:
                    st.markdown("### Thought Process")
                    st.markdown(output['thought'])
                if 'code' in output:
                    st.markdown("### Code")
                    st.code(output['code'], language="python")
                if 'output' in output:
                    st.markdown("### Output")
                    st.text(output['output'])
                else:
                    st.markdown("### Output")
                    st.text(output)
                    
        st.info("No debug information available yet. Start a conversation to see intermediate outputs.")