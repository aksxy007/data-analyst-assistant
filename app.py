import os
# os["MAX_UPLOAD_SIZE"] = 2000

import streamlit as st


st.set_page_config(layout="wide",page_title="Dashboard",page_icon=":material/dashboard:")

data_visualisation_page = st.Page(
    "./scenes/visualisation_agent.py", title="Data Visualizer", icon=":material/monitoring:"
)

pg = st.navigation(
    {
        "Visualisation Agent": [data_visualisation_page]
    }
)

pg.run()