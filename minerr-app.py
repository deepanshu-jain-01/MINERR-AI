from keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


classify_model   = load_model('Saved Models/classify-minerals.h5')
regression_model = load_model('Saved Models/calculate-mineral.h5')


st.title('MINERR - AI')
st.write("DESCRIPTION OF THE APP.........")

min_dict = {12:"Cs"}

df = pd.read_csv("Datasets/Pre-Processed-Data.csv")
scaler = MinMaxScaler()
df = scaler.fit_transform(df)



LE = LabelEncoder()
df['METALLOGEN'] = LE.fit_transform(df['METALLOGEN'])
df['LOCALITY'] = LE.fit_transform(df['LOCALITY'])
df['STATE'] =  LE.fit_transform(df['STATE'])
df['TOPOSHEET'] = LE.fit_transform(df['TOPOSHEET'])
df['HOSTROCK_TYPE1'] = LE.fit_transform(df['HOSTROCK_TYPE1'])
df['HOSTROCK_TYPE2'] = LE.fit_transform(df['HOSTROCK_TYPE2'])
df['HOSTROCK_TYPE3'] = LE.fit_transform(df['HOSTROCK_TYPE3'])
df['HOSTROCK_TYPE4'] = LE.fit_transform(df['HOSTROCK_TYPE4'])
df['MINERAL_OR'] = LE.fit_transform(df['MINERAL_OR'])

my_list=['STRATABOUND','SEDIMENTARY','BEDDED','SHEAR','CONCORDANT','DISCORDANT','RESIDUAL','LENSOID','VEIN','REMOBILISED','MAGMATIC','QUARTZ','VOLCANO']
#if not in my_list then belong to 'MORPH_OTHER'
def filter(x):
    global MORPH_STRATABOUND=0
    global MORPH_SEDIMENTARY=0
    global MORPH_BEDDED=0
    global MORPH_SHEAR=0
    global MORPH_CONCORDANT=0
    global MORPH_DISCORDANT=0
    global MORPH_LENSOID=0
    global MORPH_RESIDUAL=0
    global MORPH_VEIN=0
    global MORPH_REMOBILISED=0
    global MORPH_MAGMATIC=0
    global MORPH_QUARTZ=0
    global MORPH_VOLCANIC=0
    global MORPH_OTHER=0

    if my_list[0] in x:
        MORPH_STRATABOUND = 1
    if my_list[1] in x:
        MORPH_SEDIMENTARY = 1
    if my_list[2] in x:
        MORPH_BEDDED = 1
    if my_list[3] in x:
        MORPH_SHEAR = 1
    if my_list[4] in x:
        MORPH_CONCORDANT = 1
    if my_list[5] in x:
        MORPH_DISCORDANT = 1
    if my_list[6] in x:
        MORPH_LENSOID = 1
    if my_list[7] in x:
        MORPH_RESIDUAL = 1
    if my_list[8] in x:
        MORPH_VEIN = 1
    if my_list[9] in x:
        MORPH_REMOBILISED = 1
    if my_list[10] in x:
        MORPH_MAGMATIC = 1
    if my_list[11] in x:
        MORPH_QUARTZ = 1
    if my_list[12] in x:
        MORPH_VOLCANIC = 1
    for y in my_list:
        if y in x:
            MORPH_OTHER = 0
    else:
        MORPH_OTHER = 1

    


def user_input_features():
    METALLOGEN = st.sidebar.selectbox('METALLOGEN',('Male','Female'))
    LOCALITY = st.sidebar.selectbox('LOCALITY',('', '', '', ''))
    STATE = st.sidebar.selectbox('STATE', '')
    TOPOSHEET = st.sidebar.selectbox('TOPOSHEET', '')
    #tenure = st.sidebar.selectbox('tenure', 0.0,72.0, 0.0)
    with st.sidebar:
        HOSTROCK_TYPE1 = st.text_input('HOSTROCK_TYPE1')
        HOSTROCK_TYPE2 = st.text_input('HOSTROCK_TYPE2')
        HOSTROCK_TYPE3 = st.text_input('HOSTROCK_TYPE3')
        HOSTROCK_TYPE4 = st.text_input('HOSTROCK_TYPE4')
        MORPHOLOGY_TYPES = st.text_input("MORPHOLOGY_TYPES") 
    


    filter(MORPHOLOGY_TYPES)

    data = {'METALLOGEN':[METALLOGEN],
            'LOCALITY':[LOCALITY], 
            'STATE':[STATE], 
            'TOPOSHEET':[TOPOSHEET],
            'HOSTROCK_TYPE1':[HOSTROCK_TYPE1],
            'HOSTROCK_TYPE2':[HOSTROCK_TYPE2],
            'HOSTROCK_TYPE3':[HOSTROCK_TYPE3],
            'HOSTROCK_TYPE4':[HOSTROCK_TYPE4],
            'MORPH_STRATABOUND':[MORPH_STRATABOUND],
            'MORPH_SEDIMENTARY':[MORPH_SEDIMENTARY],
            'MORPH-BEDDED':[MORPH_BEDDED],
            'MORPH-SHEAR':[MORPH_SHEAR],
            'MORPH-CONCORDANT':[MORPH_CONCORDANT],
            'MORPH-DISCORDANT':[MORPH_DISCORDANT],
            'MORPH-LENSOID':[MORPH_LENSOID],
            'MORPH-RESIDUAL':[MORPH_RESIDUAL],
            'MORPH-VEIN':[MORPH_VEIN],
            'MORPH-MAGMATIC':[MORPH_MAGMATIC],
            'MORPH-QUARTZ':[MORPH_QUARTZ],
            'MORPH-VOLCANIC':[MORPH_VOLCANIC],
            'MORPH-OTHER':[MORPH_OTHER],           
            }


    features = pd.DataFrame(data)
    return features

input_df = user_input_features()
# df  - original data on which model is trained
# input_df - data which we have taken (input)


# Displays the user input features

st.subheader('User Input features')

#print(input_df.columns)

st.write(input_df)

# transforming our features
input_df['METALLOGEN'] = LE.transform(input_df['METALLOGEN'])
input_df['LOCALITY'] = LE.transform(input_df['LOCALITY'])
input_df['STATE'] =  LE.transform(input_df['STATE'])
input_df['TOPOSHEET'] = LE.transform(input_df['TOPOSHEET'])
input_df['HOSTROCK_TYPE1'] = LE.transform(input_df['HOSTROCK_TYPE1'])
input_df['HOSTROCK_TYPE2'] = LE.transform(input_df['HOSTROCK_TYPE2'])
input_df['HOSTROCK_TYPE3'] = LE.transform(input_df['HOSTROCK_TYPE3'])
input_df['HOSTROCK_TYPE4'] = LE.transform(input_df['HOSTROCK_TYPE4'])
#input_df['MINERAL_OR'] = LE.transform(input_df['MINERAL_OR'])
my_inputs = input_df.values
my_inputs = scaler.transform(my_inputs)
my_inputs=my_inputs.reshape(1,-1)


# Apply model to make predictions
prediction = classify_model.predict(my_inputs)
# mineral = key of the dictionary
mineral = prediction.argmax()
#prediction_proba = load_clf.predict_proba(df)
mineral_prob = prediction.max()


st.subheader('Prediction')
st.write(min_dict[mineral])

st.subheader('Prediction Probability')
st.write(mineral_prob)