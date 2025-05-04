import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from app_pages.page_upload import page_upload
from app_pages.page_deux_degres import page_deux_degres
from app_pages.page_home import page_home
from app_pages.page_team import page_team
from app_pages.page_grappes import page_grappes
from app_pages.page_sas import page_sas
from app_pages.page_pik import run_proba_inegale_interface 

st.set_page_config(
    page_title='SampleGenius by Delphin, Emmanuel, Emmanuella & Jacquelin',
    page_icon="ise.jpg")

# General formating in CSS
page_bg_img = '''
   <style>
   body {
   background-image: url("https://www.xmple.com/wallpaper/black-linear-cyan-gradient-1920x1080-c2-010506-073a47-a-120-f-14.svg");
   background-size: cover;
   color: #fff;
   }
   
   h1 {
   	color:#c4d8d6;
   }
   
   h2 {
   color : #5a6794;
   }
   
   label {
   color: #fff;
   }
   
   .stButton>button {
   color: #000000;
   background-color: #f6be65;
   font-size: large;
   }
   
   .stTextArea>label {
   color: #fff;
   font-size: medium;
   }
   
   .stTextArea>div{
   background-color: #ddddda;
   }
   
   .stTextInput>label {
   color: #fff;
   font-size: medium;
   }
   
   .stTextInput>div>div{
   background-color: #ddddda;
   }
   
   .stSelectbox>label{
   color: #fff;
   font-size: medium;
   }
   
   .stSelectbox>div>div{
   background-color: #ddddda;
   }
   
   .btn-outline-secondary{
   	background-color: #f6be65;
   }
   
   .btn-outline-secondary>span{
   	color: #000000;
   }
   
   .stAlert{
   background-color: #b0cac7;
   }
   </style>
   '''
st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu de Navigation",
            options=[
                "Accueil", 
                "Chargement des données", 
                "SAS et Stratification",  
                "Grappes", 
                "Deux degrés", 
                "Proba inégales", 
                "À propos"
            ],
            icons=[
                "house", 
                "cloud-upload", 
                "dice-5", 
                "diagram-3", 
                "shuffle", 
                "sliders", 
                "people"
            ],
            menu_icon="cast",
            default_index=0
        )

    if selected == "Accueil":
        page_home()
    elif selected == "Chargement des données":
        page_upload()
    elif selected == "SAS et Stratification":
        page_sas()
    elif selected == "Grappes":
        page_grappes()
    elif selected == "Deux degrés":
        page_deux_degres()
    elif selected == "Proba inégales":
        df = st.session_state.get("df", pd.DataFrame())  # On récupère le DataFrame stocké
        run_proba_inegale_interface(df)
    elif selected == "À propos":
        page_team()

if __name__ == "__main__":
    main()