from fastapi import FastAPI , Query 
import pandas as pd 
import numpy as np 
import re 
from sklearn.feature_extraction.text import CountVectorizer
from typing import List 
from keras.models import load_model
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware 
import google.generativeai as genai 
from contextlib import asynccontextmanager


## simple disease variables
diseaseVectorizer = CountVectorizer()
df_disease = None
df_disease_columns = None 
disease_model = None
## blood disease variables  
bloodDiseaseVectorizer = CountVectorizer()
df_blood_disease = None
df_disease_blood_columns = None 
disease_blood_model = None 
## definition of disease
df_medquad = None 
df_labels = None 
## definition of geneartive ai model 

generativeai_model = None 


def replace_space(x) : 
    return re.sub(r'\(.*?\)' , '' , x).lower().strip().lstrip().rstrip().replace(" " , "_") 



@asynccontextmanager
async def lifespan(app: FastAPI):
    global diseaseVectorizer 
    global df_disease_columns
    global disease_model 
    global df_disease 

    global bloodDiseaseVectorizer 
    global df_disease_blood_columns 
    global df_blood_disease
    global disease_blood_model

    global df_medquad
    global df_labels

    global generativeai_model 
    ## simple dieseae data prepare
    df_disease_train = pd.read_csv("Training.csv") 
    df_disease_test = pd.read_csv("Testing.csv") 
    df_disease = pd.concat([df_disease_train , df_disease_test] , axis= 0 )
    df_disease = df_disease.drop(["Unnamed: 133"] , axis=1)
    df_disease["prognosis"] = df_disease["prognosis"].apply(replace_space) 
    diseaseVectorizer.fit_transform(df_disease["prognosis"]) 
    df_disease = df_disease.drop(["prognosis"] , axis = 1) 
    df_disease_columns = df_disease.columns 
    disease_model = load_model('disease_model.h5')
    ## blood disease data prepare 
    df_blood_disease = pd.read_csv("Blood_samples_dataset_balanced_2(f).csv") 
    bloodDiseaseVectorizer.fit_transform(df_blood_disease["Disease"]) 
    df_disease_blood_columns = df_blood_disease.drop(["Disease"] , axis = 1 ).columns 
    disease_blood_model = load_model('blood_analyse_model.h5')
    ## definition 
    df_medquad = pd.read_csv("medquad.csv")
    df_medquad = df_medquad.drop(["source"] , axis=1)
    df_medquad = df_medquad.dropna() 
    df_medquad["focus_area"] = df_medquad["focus_area"].apply(lambda x : x.lower())
    ## disease targets 
    df_labels_defintions = pd.read_csv("definitions.csv")
    df_labels_pre = pd.read_csv("symptom_precaution.csv")

    df_labels = pd.merge( df_labels_defintions , df_labels_pre, on="Disease")
    df_labels["Disease"] = df_labels["Disease"].apply(replace_space)
    
    genai.configure(api_key="AIzaSyB2SICCajUAwD6pI9WzRObKGgRcg3wN5qs") 
    generativeai_model = genai.GenerativeModel(model_name="gemini-1.5-flash") 
    yield 

class BloodFeatures(BaseModel) : 
    name : str 
    value : float 


app = FastAPI(lifespan=lifespan)

origins = [
    "https://medelafia.github.io/ai-doctor/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
## endpoints for the disease predection 
@app.get("/predictDisease")
async def predict_disease(symptoms : str) : 
    new_row = {col : 0 for col in df_disease_columns }  
    for sym in symptoms.split(",") : 
        new_row[sym] = 1 
    name = diseaseVectorizer.get_feature_names_out()[disease_model.predict(np.array(list(new_row.values())).reshape(1 , -1)).argmax()]
    result = {'name' : name}
    indices = df_medquad["focus_area"].where(df_medquad["focus_area"] == name.replace("_"," ")).dropna().index
    pattern = re.compile(r"^What is .* \?$", re.IGNORECASE)
    if len(indices) > 0 : 
        for i in indices :  
            if df_medquad["question"].iloc[i].find("causes") : 
                result["causes"] = df_medquad["answer"].iloc[i]
    else : 
        result["causes"] = None 
    ## search for the disease information in the list of diseases 
    data_row = df_labels.where(df_labels["Disease"] == name).dropna()
    if data_row.size >= 1 : 
        result['definition'] = data_row.iloc[0]["Description"]
        result["precaution"] = "you have to " + " and ".join(data_row.iloc[0].values[2:].tolist())
    else : 
        ## get the data from google gemini api 
        precaution = generativeai_model.generate_content(f"what's the precautions of {name} , give me just the precautions")
        definition = generativeai_model.generate_content(f"what's {name} , give me just the definition") 
        result['definition'] = definition.text 
        result["precaution"] = precaution.text
    return result
    
## endpoints to get all diseases symptoms 
@app.get("/getSymptoms") 
async def get_symptoms() : 
    return { "symptoms" : df_disease_columns.tolist()} 

@app.post("/predictBloodDisease")
async def predict_blood_disease(blood_features : List[BloodFeatures]) : 
    new_row = { col : 0.0 for col in df_disease_blood_columns }
    for element in blood_features : 
        new_row[element.name] = element.value
    name =  bloodDiseaseVectorizer.get_feature_names_out()[disease_blood_model.predict(np.array(list(new_row.values())).reshape(1 , -1)).argmax()]
    result = {'name' : name}
    indices = df_medquad["focus_area"].where(df_medquad["focus_area"] == name).dropna().index
    pattern = re.compile(r"^What is .* \?$", re.IGNORECASE)
    if len(indices) > 0 : 
        for i in indices :  
            if re.match(pattern , df_medquad["question"].iloc[i]) : 
                result["definition"] = df_medquad["answer"].iloc[i]
            elif df_medquad["question"].iloc[i].find("causes") : 
                result["causes"] = df_medquad["answer"].iloc[i]
    else : 
        causes = generativeai_model.generate_content(f"what's the causes of {name} , give me just the causes")
        definition = generativeai_model.generate_content(f"what's {name} , give me just the definition")  
        result["causes"] = causes.text 
        result["definition"] = definition.text
    return result 

@app.get("/getBloodFeatures") 
async def getBloodFeatures() : 
    return { 'features' : df_disease_blood_columns.tolist() } 