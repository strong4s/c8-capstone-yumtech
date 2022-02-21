

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

import math 

# Directory
DATA_csv = "data/"
DATA_png = "data/__img/"
URL = "data/URL.csv"


class Recommender_Page():
    def __init__(self):
        self.model = Recommender_Base()
        self.url = pd.read_csv(URL)
        self.start()

    # Intro txt
    def display_intro(self):
        st.title('Recommender Engine')
        # st.markdown(" - This recommender engine explores our Artist's closely related popular tracks from the top 200 charts.")
        # st.markdown(" - The goal of this project is to gain insights on potential artist collaborations, explore suitable music styles/ genres, as well as obtain baseline audio features as a guide for track design.")
        # st.markdown(" - Samples could be increased to gain better results.")
        st.markdown("## Input Ingredients")

    # Display Recommendation Info
    def display_recom_info(_self, df):
        #seed_service = SeedService()
        st.markdown("# Results")
    
    def make_clickable(self,name, link):
        # target _blank to open new window
        # extract clickable text to display for your link
        text = name
        return f'<a target="_blank" href="{link}">{text}</a>'

    def start(self):
        # Display Intro
        self.display_intro()


        ingredients = st.text_input(
        "Enter ingredients you would like to cook with (seperate ingredients with a comma)",
        "tomato, lettuce, cheese, sausage, pepper, catsup",
        )

        # Item count
        items = st.slider("No. of Results", 1, 100, key="items", value=20)

        if st.button("Submit"):
            status_msg = "Generating recommendations for {0} - {1}.."
            st.text(status_msg)
            self.model.main(ingredients,items)
            st.markdown("# Results")
            st.markdown("### Course for this combination: **{0}**".format(self.model.course))
            st.markdown("### Region: **{0}**".format(self.model.region))
            #st.write(self.model.recommendations)
            cols = ["title","cuisine","course","region","continent","country","sugar_g","sodium_g","fat_g","fiber_g","protein_g",
                "energy_kcal", "sugar_class", "sodium_class","fat_class","fiber_class", "protein_class","energy_class","url"]
            recom_df = self.model.df[self.model.df.recipe_id.isin(self.model.recommendation_ids)]
            recom_df = recom_df.merge(self.url,on="recipe_id",how="left")
            recom_df["link"] = recom_df.apply(lambda row: self.make_clickable(recom_df["Source"], row["url"]), axis=1)
            st.write(recom_df[cols])
            
        

        # # Load data
        # df = self.load_csv(mode="all")

        # # ======================== #
        # # Create Handlers

        # # Get artist_name
        # df_artist = df.drop_duplicates(subset = "artist_name")[["artist_name", "artist_id"]]
        # artist_name = st.selectbox("Select Artist: ", df_artist, key="artist_name")
        # del df_artist

        # # Get track_name
        # df_tracks = df[df.artist_name == artist_name][["track_name", "track_id"]]
        # track_name = st.selectbox("Select Track: ", df_tracks, index=0, key="track_name")
        # del df_tracks

        # # Query track_id
        # track_id = df[(df.artist_name == artist_name) & (df.track_name == track_name)].track_id.squeeze()
        # # ======================== #

        # # Display track info
        # self.display_track_info(track_id)        

        # # Item count
        # items = st.slider("Select no. of items", 1, 100, key="items", value=10)
        # # Method
        # method = st.radio("Select metric:", ["cosine_dist", "manhattan_dist", "euclidean_dist"], key="method")

        # # Obtain recommendations
        # status_msg = ""
        # if st.button("Submit"):
        #     status_msg = "Generating recommendations for {0} - {1}..".format(track_name, artist_name)
        #     st.text(status_msg)
        #     recom_df = self.generate_by_track_id(track_id, items, method).recommendations

        #     # Display results
        #     self.display_recom_info(recom_df)

        #     # Outro message
        #     st.markdown("# So, what do you think? \n### Can **{0}** make a comeback?".format(artist_name))
        #     st.markdown(" - Try increasing the number of items for better results!")
        #     st.markdown(" - Different distance metrics also yield different outcomes!")
        #     st.markdown("### Who do you think should make a comeback next?")

        #     #st.balloons()
        #     del recom_df

        # del df
        # return