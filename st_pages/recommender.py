

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


class Recommender_Page():
    session_state = None
    status= None

    def __init__(self, mode="ingredient"):
        self.mode = mode
        self.model = Recommender_Base()
        #self.init_state()

        self.start()
        

    # Intro txt
    def display_intro(self):
        st.title('Recommender Engine')
        st.markdown("#### Philippine dietary reference intake labels:")
        st.markdown("Below are class labels used to classify intake of sugar, sodium, fat, and fiber.")
        st.markdown("- **0** - low, 5 servings doesn't exceed PDRI.")
        st.markdown("- **1** - adequate, up to 3 servings only")
        st.markdown("- **2** - up to 2 servings only")
        st.markdown("- **3** - single serving only")
        st.markdown("- **4** - high, exceeds PDRI on a single serving")
        

        st.markdown("## Input Ingredients")

    # Display Recommendation Info
    def display_recom_info(_self, df):
        st.markdown("# Results")
    
    def make_clickable(self,name, link):
        text = name
        return f'<a target="_blank" href="{link}">{text}</a>'

    def start_recipe(self):
        st.title('Recommender Engine')
        method = st.radio("Select recipes:", ["All", "FNRI"], key="method")
        if method == "All":
            region_ls = self.model.le_region.classes_
            region_sel = st.selectbox("Select Track: ", region_ls, index=0)
        else:
            df = self.model.df[self.model.df.recipe_id.str.contains("fnri")]
            st.write(df)
        pass



    def submit_ingreds(self, ingredients, filter = None):
        # Create Message
        status_msg = "Generating recommendations for %s.." % ingredients
        st.text(status_msg)
        # Launch model
        self.model.main(ingredients,items=None,filter=filter)
        # Print results
        st.markdown("# Results")
        st.markdown("### Course: **{0}**".format(self.model.course))
        st.markdown("### Region: **{0}**".format(self.model.region))

        #st.write(self.model.recommendations)
        cols = ["title","cuisine","course","region","continent","country","sugar_g","sodium_g","fat_g","fiber_g",
            "sugar_class", "sodium_class","fat_class","fiber_class", "Source","url","cosine_dist", "bow_str"]
        recom_df = self.model.df[self.model.df.recipe_id.isin(self.model.recommendation_ids)]
        recom_df["bow_str"] = recom_df.bow_str.apply(lambda x: x.replace(" ",", "))

        recoms = self.model.recommendations[["recipe_id","cosine_dist"]]
        recom_df =  recom_df.merge(recoms,how="left",on="recipe_id")
        recom_df = recom_df.merge(self.model.url,on="recipe_id",how="left")
        recom_df["link"] = recom_df.apply(lambda row: self.make_clickable(self.model.url["Source"], row["url"]), axis=1)

        return recom_df[cols]

    def init_state(self):
        if 'load state' not in st.session_state:
            st.session_state.load_state = False
            st.session_state.index=0

    def display_results(self):
        if st.session_state.load_state == True:
            #index_ls = self.model.df.head().shape[0]
            col1, col2 = st.columns([2,2])
            next = st.button(label='Next')
            prev = st.button(label="Prev")
            with col1:
                if next:
                    st.session_state.index+=1
            with col2:
                if prev:
                    st.session_state.index-=1

            st.write(st.session_state.index)
            pass
        st.write(st.session_state.load_state)


    # --------------------------------------------------------------------------
    # Ingredient Form
    # --------------------------------------------------------------------------
    #@st.cache(suppress_st_warning=True)
    def ingredient_form(self):
        with st.form(key='ingredient_input'):
            # Ingredients Pane
            ingredients = st.text_input(
            "Enter ingredients (seperate ingredients with a comma)",
            "tomato, lettuce, cheese",
            )

            cuisine_ls = ["all"]+self.model.df.cuisine.unique().tolist()
            cuisine = st.selectbox("Select Cuisine: ", cuisine_ls)

            course_ls = ["all"]+self.model.df.course.unique().tolist()
            course = st.selectbox("Select Course: ", course_ls)

            # Submit button
            submit_input = st.form_submit_button(label='Submit')

        if submit_input:
            # Create filters
            filter = {}
            # Cuisine
            if cuisine != "all":
                filter["cuisine_label"] = self.model.le_cuisine.transform([cuisine])[0]
            if course != "all":
            # Course
                filter["course_label"] = self.model.le_course.transform([course])[0]
            
            return [ingredients, filter]


    # =============================================================================
    # * Start Page
    # =============================================================================
    #@st.cache(suppress_st_warning=True)
    def start(self):

        # Display Intro
        self.display_intro()
        # --------------------------------------------------------------------------
        # Process Recommendations
        # --------------------------------------------------------------------------

        get = self.ingredient_form()
        if get is not None:
            ingredients = get[0]
            filter = get[1]
        else:
            return #print("Error")


        if ingredients:
            df = self.submit_ingreds(ingredients, filter)
            st.write(df)
            self.stats = "asas"



        # # --------------------------------------------------------------------------
        # # Navigate Results
        # # --------------------------------------------------------------------------
        # if recom_df is not None:
            
        #     index_ls = recom_df.shape[0]
        #     col1, col2 = st.columns([2,2])

        #     next = st.button(label='Next')
            

        #     with col1:
        #         if next:
        #             st.session_state["index"]+=1

        #     if st.session_state["index"] > 0:
        #         prev = st.button(label='Prev')
        #         with col2:
        #             if prev:
        #                 st.session_state["index"]+=1
                
        #     st.write(st.session_state["index"])
        #     st.stop()
        #     # cuisine_ls = ["All"]+recom_df.cuisine.unique().tolist()   # +self.model.le_cuisine.classes_.tolist()
        #     # cuisine = st.selectbox("Select Cuisine: ", cuisine_ls, index=0)
        #     # print(cuisine)
        #     # if cuisine == "All":
        #     #     title_ls = recom_df.title.tolist()
        #     #     title = st.selectbox("Select Recipe: ", title_ls, index=0)
