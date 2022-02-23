#==============================================================================#
#   Title: Base Recommender Engine
#   Author: Kui
#   Date: 2020-2-14
#------------------------------------------------------------------------------#
#   Version 1.1
#------------------------------------------------------------------------------#
#    2020-2-21 - Version 1.1   
#    2020-2-14 - Version 1.0
#------------------------------------------------------------------------------#

#==============================================================================#
#   Introduction
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#   SETTINGS
#------------------------------------------------------------------------------#

REGION_MODEL = "data/models/SVC_clf_rbf.mdl"
COURSE_MODEL = "data/models/LSTM_optimal.mdl"

REGION_PIPE = "data/models/5.Vectorizer-PCA.mdl"
LE_VECT = "data/models/le_vect.mdl"
LE_REGION = "data/models/le-region.mdl"
LE_CUISINE = "data/models/le-cuisine-v2.mdl"
LE_CONTINENT = "data/models/le-continent.mdl"
LE_COURSE = "data/models/le-courses-2.mdl"

SCALER = "data/models/categorical_scaler.mdl"

CORPUS_REG = "data/models/corpus-region.mdl"
CORPUS_COR = "data/models/corpus-courses-v1.mdl"

DF = "data/all_scaled-compact-v1.mdl"
IDF = "data/models/corpus-courses-v2.mdl"
URL_DF = "data/URL.csv"

#------------------------------------------------------------------------------#
#   Import packages
#------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import joblib as j
from os.path import exists

import re, sys

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# from models.ravel import Ravel
from ravel import Ravel
# from models.encode import Encoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

from fuzzywuzzy import fuzz, process


#==============================================================================
# ** Recommender_Base
#------------------------------------------------------------------------------
#   This is the base class of the recommender engine.
#------------------------------------------------------------------------------
class Recommender_Base:
    #-------------------------------------------------------------
    # * Create Variables
    #-------------------------------------------------------------
    region_model        = None
    region_pipe         = None
    course_model        = None
    course_pipe         = None
    corpus_courses      = None
    corpus_region       = None
    idf                 = None

    le_region           = None
    le_cuisine          = None
    le_continent        = None
    le_course           = None

    df                  = None
    recommendations     = None
    recommendation_ids  = None


    #-------------------------------------------------------------
    # * Initialize
    #-------------------------------------------------------------
    def __init__(self):
        #pd.set_option("max_columns", 999)
        #pd.set_option("max_rows", 999)
        self.load_models()

    #-------------------------------------------------------------
    # * Clean Text
    #-------------------------------------------------------------
    def clean_text(self,txt):
        txt = txt.split(",")
        # Attempt to clean
        ntxt = []
        for w in txt:
            w = re.sub("^(\s+)|(\s+)$","",w).lower()
            w = [i for i in w.split(" ") if len(i) >= 3]
            w = " ".join(w)
            ntxt.append(w)
        # Re-assign
        txt = ntxt
        txt = [i.replace(" ","_") for i in txt]
        # Sort
        txt = sorted(txt)
        # Return
        return txt

    #-------------------------------------------------------------
    # * Course Input Vectorizer
    #-------------------------------------------------------------
    def course_input_vectorizer(self,keys):
        # Explode compound words:
        get = []
        for k in keys:
            if "_" in k: get += (k.split("_"))
            elif " " in k: get += (k.split(" "))  
            else: get.append(k)
        # Clean
        get = [re.sub("^(\s+)|(\s+)$","",i) for i in get]
        # Get intersection
        get = set(keys).intersection(set(self.le_vect.classes_))
        get = sorted(list(get))
        get=pd.DataFrame({"n":[get]})

        vect = get.n.apply(lambda x: self.le_vect.transform(x))
        #vect = self.le_vect.transform(get)
        # pad sequences
        vect = pad_sequences(vect,130+1)

        return vect

    #-------------------------------------------------------------
    # * Region Input Vectorizer
    #-------------------------------------------------------------
    def region_input_vectorizer(self, keys):
        # text_string = [" ".join(text_arr)]
        # print(text_string)
        get = set(keys).intersection(self.corpus_region)
        get = sorted(list(get))
        get = " ".join(get)
        get = [get]

        # Tf-IDF Vectorizer upto PCA
        vect = self.region_pipe.transform(get)
        return vect

    #-------------------------------------------------------------
    # * Load Models
    #-------------------------------------------------------------
    def load_models(self):
        self.region_model = j.load(REGION_MODEL)
        self.region_pipe = j.load(REGION_PIPE)

        # self.df = pd.DataFrame()
        self.le_vect = j.load(LE_VECT)
        self.course_model = j.load(COURSE_MODEL)
        self.df = j.load(DF)

        # Label Encoders
        self.le_region=j.load(LE_REGION)
        self.le_course=j.load(LE_COURSE)
        
        self.le_continent=j.load(LE_CONTINENT)
        self.le_cuisine=j.load(LE_CUISINE)

        # Load Corpus
        self.corpus_courses = set(j.load(CORPUS_COR).key.tolist())
        self.corpus_region = set(j.load(CORPUS_REG).key.tolist())

        self.url = pd.read_csv(URL_DF)
        self.url["recipe_id"] = self.url.recipe_id.astype("string")

        self.sc = j.load(SCALER)
        self.idf = j.load(IDF)

    #-------------------------------------------------------------
    # * Calculate hits
    #-------------------------------------------------------------
    def calculate_hits(self, df_keys={}, user_keys={}):
        lookup = self.idf

        #a=pd.Series([i for i in list(df_keys) if "_" not in i]).unique().tolist()
        a=set(df_keys) #(a)
        b=set(user_keys)
        
        hits = a.intersection(b)

        try: a=lookup[lookup.key.isin(hits)].idf.sum() 
        except: a=0
        if a == np.nan: a=0

        try: b=lookup[lookup.key.isin(set(user_keys))].idf.sum() 
        except: b=0
        if b == np.nan: b=0


        dist = (a/b)
        return dist

    #-------------------------------------------------------------
    # * Classify
    #-------------------------------------------------------------
    def classify(self, text_arr=None, type="region"):
        if text_arr is None: return
        if type == "region":
            vect = self.region_input_vectorizer(text_arr)
            clas = self.region_model.predict(vect)
            return clas
        elif type == "course":
            vect = self.course_input_vectorizer(text_arr)
            pred = self.course_model.predict(vect)
            clas = np.argmax(pred,axis=1)[:].ravel()[0]
            return clas
            # for i,v in enumerate(clas):
            #     if v > 0: return i

    #-------------------------------------------------------------
    # * Main
    #-------------------------------------------------------------
    def main(self, text="tomato lettuce cheese", items=10, filter=None):
        
        if "*" not in text:
            # Clean text
            text = self.clean_text(text)

        # ------------------------------------------------
        # Classify Input Region & Course
        # ------------------------------------------------
        region_label = self.classify(text,type="region")
        if 'course_label' in filter.keys(): 
            course_label = filter["course_label"]
        else:
            course_label = self.classify(text,type="course")
        # Get Input Region & Course Labels
        region_label = [region_label[0]]
        course_label = [course_label]
        # Assign values to self
        self.region = self.le_region.inverse_transform(region_label)[0]
        self.course = self.le_course.inverse_transform(course_label)[0] 

        # ------------------------------------------------
        # Generate Seed
        # ------------------------------------------------
        # Reset if * in text:
        if "*" in text:
            self.region = "all" 
            df={"course_label": course_label,}
        else:
            df = {
            "region_label": region_label,
            "course_label": course_label,
            }

        # Set defaults to adequate
        df["sugar_class"] = 1
        df['sodium_class'] = 1
        df['fat_class'] = 1
        # Get closest to high fiber!
        df['fiber_class'] = 2
        # Create seed DF
        df = pd.DataFrame(df)

        # Start Predict
        self.predict(df, items=items, text=text, filter=filter)

        print(self.recommendations)

    #-------------------------------------------------------------
    # * Predict
    #-------------------------------------------------------------
    def predict(self, df, items=None, filter=None, text=None):
        
        # ------------------------------------------
        # Scale values
        # ------------------------------------------
        cols = df.columns.tolist()
        for col in cols:
            df[col+"_sc"] = self.sc.transform(df[[col]])
        # Update columns
        cols = [i+"_sc" for i in cols]
        metric = "cosine_dist"

        # ------------------------------------------
        # Filter Data
        # ------------------------------------------
        get_df = self.df.copy()
        if filter is not None:
            for k in filter.keys():
                get_df = get_df[get_df[k] == filter[k]]
            if get_df.shape[0] == 0: return None
        

        # ------------------------------------------
        # Apply thresholds
        # ------------------------------------------
        get_df = get_df[get_df.sugar_class < 3]
        get_df = get_df[get_df.sodium_class < 3]
        get_df = get_df[get_df.fat_class < 3]
        if get_df.shape[0] == 0: return None

        print(get_df.shape)

        # ------------------------------------------
        # Get Fuzzy distance
        # ------------------------------------------
        if text is not None:
            if "*" in text:
                get_df["sim"] = 1
            else:
                get_df["sim"] = get_df.bow_str.apply(lambda x: fuzz.partial_ratio(x, " ".join(text)))
                get_df["sim"] = get_df["sim"] / 100
                get_df = get_df[get_df.sim > 0.6]
                print(get_df.shape)

        # Update to df, closest to 1
        df["sim"] = 1
        if get_df.shape[0] == 0: return None

        # Get hits
        if "*" in text:
            pass
        else:
            get_df["hits"] =  get_df.bow.apply(lambda x: self.calculate_hits(x, text))
            get_df = get_df[get_df.hits > 0.6]
            df["hits"] = 1
            print(get_df.hits.shape)
            cols.append("hits")
            if get_df.shape[0] == 0: return None
        

        #return
        # ------------------------------------------
        # Calculate similarity
        # ------------------------------------------
        get_df[metric] = get_df.apply(lambda x: 1 - cosine_similarity(x[cols].values.reshape(1, -1), df[cols].values.reshape(1, -1)).mean(), axis=1)

        
        # get_df['cosine_dist'] = get_df.apply(lambda x: 1-cosine_similarity(x[cols].values.reshape(1, -1), \
        #     df_seed)\
        #     .flatten()[0], axis=1)
    
        # Rank
        recommendation_df = get_df.sort_values(by=[metric, "fiber_class"], ascending=[True, False])
        if items is not None: recommendation_df = recommendation_df.head(items)
        recommendation_df = recommendation_df[['title','recipe_id','region','cuisine','continent']+cols+[metric]]
        self.recommendations = recommendation_df
        self.recommendation_ids = recommendation_df['recipe_id'].values#.tolist()

# from tensorflow.random import set_seed
# set_seed(41)

def main():
    m = Recommender_Base()
    m.main("chicken, soy sauce, garlic, tofu")


# main()
