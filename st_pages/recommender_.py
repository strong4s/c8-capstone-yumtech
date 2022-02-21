
from services.recommender_service import RecommenderService
from services.seed_service import SeedService

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib as j
# pd.set_option("max_columns", 999)

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


import math 

# Directory
DATA_csv = "db/data/"
DATA_png = "db/png/"

# Query tracks from CSV file
CSV = ["abra.csv", "top200-genres.csv"]
RAW = ["top200.csv"]
ART = ["album_art.csv", "artist_art.csv"]

class Recommender_Page():
    def __init__(self):
        self.start()

    # Intro txt
    def display_intro(self):
        st.title('Recommender Engine')
        st.markdown(" - This recommender engine explores our Artist's closely related popular tracks from the top 200 charts.")
        st.markdown(" - The goal of this project is to gain insights on potential artist collaborations, explore suitable music styles/ genres, as well as obtain baseline audio features as a guide for track design.")
        st.markdown(" - Samples could be increased to gain better results.")
        st.markdown("## Select Track")



    # Display Recommendation Info
    def display_recom_info(_self, df):
        seed_service = SeedService()

        #data = df.copy()
        st.markdown("# Results")

        # 1. Display Potential Colaborators
        artists = df.artist_name.value_counts()
        artists.columns=["count"]

        st.markdown("### Potential Collaborations/ Artists Styles to Explore")
        st.write(artists)

        # Get Artist Images
        art_df = pd.read_csv(DATA_csv+ART[1])
        art_df = art_df[art_df.track_id.isin(df.track_id.tolist())]
        img = []
        nlist = [df[df.artist_name == i].head(1).track_id.squeeze() for i in artists.index.tolist()]#data.track_id.tolist()
        for track_id in nlist:
            i = art_df[art_df.track_id == track_id].head(1).image.squeeze()
            if type(i) == str: img.append(i)

        # Display Artist Images
        st.image(img, width=90)
        st.markdown(" - A generated list of artists may be possible collaborators, as well as a resource for experimenting with their popular music styles while creating new tracks.")

        # clear vars
        del artists, art_df, img, nlist
        
        # 2. Genres to Explore
        st.markdown("### Explore new Genres!")
        genres_df = df.predicted_genre.value_counts().to_frame().T
        genres = {"funky": "Funk", "hard rock": "Hard rock", "hiphop": "Hiphop", "pop song": "Pop", "rnb": "R&B"}
        # Fill NA
        for col in genres.keys():
            if col not in genres_df.columns.tolist():
                genres_df[col] = 0
        # Rename cols
        genres_df.rename(columns = genres, inplace=True)
        
        # Show Genre barplot
        fig, ax = plt.subplots(figsize=(12,3))
        ax = sns.barplot(data = genres_df)
        plt.title("Genre Frequency Results",size=15)
        st.pyplot(fig)

        st.markdown(" - The Artist can explore the following genres, which closely suits their style in writing new songs!")
        st.markdown(" - The recommender generated a list of genres closest to the seed track, obtained from the top 200 charts.")
        
        # clear vars
        del genres_df, fig, ax
        
        # 4. Designing New Beats
        st.markdown("### Design new Beats!")
        
        # Query raw values
        raw_df = pd.read_csv(DATA_csv+RAW[0])
        raw_df.drop_duplicates("track_name", inplace=True)
        raw_df = raw_df[raw_df.track_id.isin(df.track_id.tolist())]

        # Plot Raw Stats
        nplots = len(seed_service.feature_cols)
        fig, axs = plt.subplots(3,3, figsize = (12,6))

        for i in range(3):
            for j in range(3):
                col_name = seed_service.feature_cols[i*3+j]
                get = raw_df[col_name]
                x = sns.histplot(data = get, ax = axs[i,j], kde=True, bins=10)
                
                axs[i,j].set_title(col_name)
                axs[i,j].set(xlabel='')

                del get, x

        plt.subplots_adjust(
                        #left=0.9,
                        #bottom=0.7, 
                        #right=0.9, 
                        #top=0.9, 
                        wspace=0.4, 
                        hspace=0.9)
        st.pyplot(fig)

        st.markdown(" - When producing a new track, the following audio features like *tempo* can provide ideas as a baseline for beats design.")
        st.markdown(" - These data were extracted from the top recommendation results.")
        st.markdown("### Recommender Top Tracks: ")
        
        st.write(df.drop(["track_id"], axis=1).reset_index(drop=True))

        del df, raw_df, fig, axs, seed_service


    # Display Track Info
    def display_track_info(self, track_id):
        seed_service = SeedService()
        seed_service.get_seed([track_id])
        data = seed_service.seed.data
        
        statcols = [i for i in data.columns.tolist() if "_prob" in i]
        stats = data[statcols].drop(["all_genre_prob","predicted_genre_prob"], axis=1)
        stats.columns = ["Funk", "Hard rock", "Hiphop", "Pop", "R&B"]

        # Print track info
        st.markdown("#### Track information")
        art_df = self.load_csv(mode="art")
        art_df = art_df[art_df.track_id == track_id]
        img = art_df.head(1).image.squeeze()
        st.image(img, width=300)

        text = "*Track name:* {0}\n".format(data.track_name.squeeze())
        text += "\n*Artist:* {0}\n".format(data.artist_name.squeeze())

        # Genre
        genre = data.predicted_genre.squeeze().replace("rnb","r&b")
        genre = [i for i in stats.columns.tolist() if i.lower() in genre][0]
        text += "\n*Classified genre:* {0}\n".format(genre)
        st.markdown(text)

        # Print genre stats
        fig, ax = plt.subplots(figsize = (10,3))
        plt.gca().set_ylim(0,1)
        plt.title("Genre Probabilities for \"{0}\"".format(data.track_name.squeeze()),size=15)
        ax = sns.barplot(data=stats)
        st.pyplot(fig)

        del data, seed_service, statcols, stats, genre, fig, ax, art_df, img


        
    # @st.cache(suppress_st_warning=True)
    def generate_by_track_id(_self, track_id, items=20, method = "cosine_dist"):
        # Create SeedService
        seed_service = SeedService()
        # Create RecommenderService
        recom_service = RecommenderService()

        # Generate Seed track data, See seed_service.py
        seed_service.get_seed(q=[track_id], col=["track_id"])
        
        # Generate recommendations
        recom_service.generate(seed_service.seed, method, items)

        del seed_service
        return recom_service

    @st.cache(allow_output_mutation=True)
    def load_csv(_self, mode="all"):
        if mode == "all":
            df = pd.concat((pd.read_csv(DATA_csv+i).sort_values("artist_name") for i in CSV))
            df.drop_duplicates(["track_name"], inplace=True)
            # Remove duplicate artist: Abra
            # df = df[df.artist_id != "5mNum7eUoqWS6NBo91NYHP"] 
            # Remove other columns
            df = df.drop(["playlist_id", "playlist_name"], axis=1)
        elif mode == "art":
            df = pd.read_csv(DATA_csv+ART[0])

        return df

    def start(self):
        # Display Intro
        self.display_intro()

        # Load data
        df = self.load_csv(mode="all")

        # ======================== #
        # Create Handlers

        # Get artist_name
        df_artist = df.drop_duplicates(subset = "artist_name")[["artist_name", "artist_id"]]
        artist_name = st.selectbox("Select Artist: ", df_artist, key="artist_name")
        del df_artist

        # Get track_name
        df_tracks = df[df.artist_name == artist_name][["track_name", "track_id"]]
        track_name = st.selectbox("Select Track: ", df_tracks, index=0, key="track_name")
        del df_tracks

        # Query track_id
        track_id = df[(df.artist_name == artist_name) & (df.track_name == track_name)].track_id.squeeze()
        # ======================== #

        # Display track info
        self.display_track_info(track_id)        

        # Item count
        items = st.slider("Select no. of items", 1, 100, key="items", value=10)
        # Method
        method = st.radio("Select metric:", ["cosine_dist", "manhattan_dist", "euclidean_dist"], key="method")

        # Obtain recommendations
        status_msg = ""
        if st.button("Submit"):
            status_msg = "Generating recommendations for {0} - {1}..".format(track_name, artist_name)
            st.text(status_msg)
            recom_df = self.generate_by_track_id(track_id, items, method).recommendations

            # Display results
            self.display_recom_info(recom_df)

            # Outro message
            st.markdown("# So, what do you think? \n### Can **{0}** make a comeback?".format(artist_name))
            st.markdown(" - Try increasing the number of items for better results!")
            st.markdown(" - Different distance metrics also yield different outcomes!")
            st.markdown("### Who do you think should make a comeback next?")

            #st.balloons()
            del recom_df

        del df
        return