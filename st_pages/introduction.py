import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib as j
# pd.set_option("max_columns", 999)

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from recommender_model import Recommender_Base

import streamlit.components.v1 as components  # Import Streamlit

from PIL import Image

# Directory
DATA_csv = "data/"
DATA_png = "data/__img/"
DATA_assets = "assets/"
CSV="dum.csv"

NUM = "60,983"

class Introduction_Page:
    def __init__(self):
        self.main()
        pass

    def display_intro(self):
        st.markdown("# Go Healthy!")

        st.image("https://www.nrh.ie/wp-content/uploads/2020/07/fresh-fruit-and-vegetables.jpg")

        st.markdown(" Go Healthy! is a healthier meals recommender designed for less sugar, sodium, and fat consumption, while still providing high-fiber meal options possible.")
        st.markdown(" Currently our database has *{0}* recipes for you to explore!".format(NUM))

        #st.image(DATA_assets+"loadings.png")

        st.markdown(" # Explore Recipes")
        st.markdown(" Explore global cuisines from our database!")
        explore = st.selectbox("Select: ",["Low Sugar", "Low Sodium", "Low Fat", "High Fiber"], index=3)
        st.markdown("## %s Recipes" % explore)
        if explore == "Low Sugar":
            get = "8"
        elif explore == "Low Sodium":
            get = "10"
        elif explore == "Low Fat":
            get = "12"
        elif explore == "High Fiber":
            get = "14"

        html = '<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/~Kai-03/%s.embed"></iframe>' % get
        components.html(html,width=900,height=800)

    def main(self):
        self.display_intro()


    