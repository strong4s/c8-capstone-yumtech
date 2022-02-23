import streamlit as st

from st_pages.recommender import Recommender_Page
from st_pages.introduction import Introduction_Page

def initialize():
    PAGES = [
        'Introduction',
        'Go Healthy!'
    ]
    page = st.sidebar.radio("Page Navigation", PAGES)

    if page == 'Go Healthy!':
        on = Recommender_Page()
    elif page == 'Introduction':
        on = Introduction_Page()
    
        


if __name__ == "__main__":
    initialize()
