from keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler



classify_model   = load_model('Saved Models/classify-minerals.h5')
regression_model = load_model('Saved Models/calculate-mineral.h5')


st.title('MINERR - AI')
st.write("DESCRIPTION OF THE APP.........")

min_dict = {24: 'Fe-Hematite', 13: 'Cu', 1: 'Au', 0: 'Al-Bauxite', 36: 'Pb-Zn', 29: 'Mn', 30: 'Mn-Fe', 10: 'Cr', 27: 'Fe-Ti-V', 9: 'Be-Nb-Ta', 18: 'Cu-Pb', 23: 'Cu-Zn', 26: 'Fe-Magnetite', 40: 'WO3', 38: 'Pb-Zn-Cu', 37: 'Pb-Zn-Ag', 25: 'Fe-Hematite-Mn', 20: 'Cu-Pb-Zn', 6: 'Au-W', 3: 'Au-Cu', 35: 'Pb', 34: 'Nb-Ta-Li-Sn', 5: 'Au-Mo', 19: 'Cu-Pb-Ba', 12: 'Cs', 41: 'Zn', 4: 'Au-Cu-Zn', 21: 'Cu-Pb-Zn-Sb-Py', 8: 'Be', 11: 'Cr-PGE', 39: 'U', 33: 'Nb-Ta', 17: 'Cu-Ni', 32: 'Mo-U-Cu', 14: 'Cu-Co', 16: 'Cu-Mo-Au', 2: 'Au-Ag-Cu-Pb-Zn', 28: 'Ma', 31: 'Mo', 15: 'Cu-Fe-Ti-V', 7: 'Ba'}

df = pd.read_csv("Datasets/Pre-Processed-Data.csv")
df = df.drop("RESERVE_AMT",axis=1)
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

ros=RandomOverSampler()
X,Y=ros.fit_resample(df,df["MINERAL_OR"])

scaler = MinMaxScaler()
df = scaler.fit_transform(X)

my_list=['STRATABOUND','SEDIMENTARY','BEDDED','SHEAR','CONCORDANT','DISCORDANT','RESIDUAL','LENSOID','VEIN','REMOBILISED','MAGMATIC','QUARTZ','VOLCANO']
#if not in my_list then belong to 'MORPH_OTHER'
def filter(x):
    global MORPH_STRATABOUND
    global MORPH_SEDIMENTARY
    global MORPH_BEDDED
    global MORPH_SHEAR
    global MORPH_CONCORDANT
    global MORPH_DISCORDANT
    global MORPH_LENSOID
    global MORPH_RESIDUAL
    global MORPH_VEIN
    global MORPH_REMOBILISED
    global MORPH_MAGMATIC
    global MORPH_QUARTZ
    global MORPH_VOLCANIC
    global MORPH_OTHER

    if my_list[0] in x:
        MORPH_STRATABOUND = 1
    else:
        MORPH_STRATABOUND = 0
    if my_list[1] in x:
        MORPH_SEDIMENTARY = 1
    else:
        MORPH_SEDIMENTARY = 0
    if my_list[2] in x:
        MORPH_BEDDED = 1
    else:
        MORPH_BEDDED = 0
    if my_list[3] in x:
        MORPH_SHEAR = 1
    else:
        MORPH_SHEAR = 0
    if my_list[4] in x:
        MORPH_CONCORDANT = 1
    else:
        MORPH_CONCORDANT = 0
    if my_list[5] in x:
        MORPH_DISCORDANT = 1
    else:
        MORPH_DISCORDANT = 0
    if my_list[6] in x:
        MORPH_LENSOID = 1
    else:
        MORPH_LENSOID = 0
    if my_list[7] in x:
        MORPH_RESIDUAL = 1
    else:
        MORPH_RESIDUAL = 0
    if my_list[8] in x:
        MORPH_VEIN = 1
    else:
        MORPH_VEIN = 0
    if my_list[9] in x:
        MORPH_REMOBILISED = 1
    else:
        MORPH_REMOBILISED = 0
    if my_list[10] in x:
        MORPH_MAGMATIC = 1
    else:
        MORPH_MAGMATIC = 0
    if my_list[11] in x:
        MORPH_QUARTZ = 1
    else:
        MORPH_QUARTZ = 0
    if my_list[12] in x:
        MORPH_VOLCANIC = 1
    else:
        MORPH_VOLCANIC = 0
    for y in my_list:
        if y in x:
            MORPH_OTHER = 0
            break
        else:
            MORPH_OTHER = 1

    


def user_input_features():
    METALLOGEN = st.sidebar.selectbox('METALLOGEN',('-', 'ADASH RAIMAL BELT', 'AGNIGUNDLA BELT', 'AHIRWALA-BALESWAR',
                'AHIRWALA-BALESWAR BELT', 'AKOLA-WARI-BHIDER BELT','AKOLA-WARI-BHINDER BELT', 'ALADAHALLI BELT', 'ALWAR BELT',
                'BABABUDAN BELT', 'BABABUDAN RANGE', 'BAILADILA BELT', 'BALAGHAT BELT', 'BALDA-DEWAKA BERA BELT',
                'BALDA-DEWAKI BERA BELT', 'BARAGANDA- PARASIA BELT', 'BASTAR-MALKANGIFII PEGMATITE', 'BASTAR-MALKANGIRI PEGMATITE',
                'BAULA-NAUSAHI BELT', 'BELLARY (SANDUR SCHIST BELT)','BIHAR MICA BELT', 'BONAI-BADAMPAHAR BELT',
                'BONAI-KEONJHAR RANGE BELT', 'BONAI-NOAMUNDI-JAMDA BELT','CHIRIA-MANOHARPUKUR SECTOR', 'CHITAR-KALABAR BELT',
                'CHITRADURGA BELT', 'CHOTANAGPUR GNEISSIC COMPLEX PROVINCE', 'DERI-AMBAMATA BELT', 'DERI-AMBAMATA BELT(EASTERN SECTOR)',
                'DERI-AMBAMATA BELT(WESTERN SECTOR)', 'EAST COAST PROVINCE', 'EASTERN GHAT BELT', 'GADAG SCHIST BELT', 'GANI KALVA BELT',
                'GARUMAHISANI PROVINCE', 'GARUMAHISANI PROVINCE (RAIRANGPUR BELT)','GHATTIHOSAHALLI BELT', 'GHUGRA-KAYAR BELT',
                'GOA Fe - Mn PROVINCE', 'GODAVARI RIFT BELT', 'GRANULITE BELT OF KERALA', 'HARUR-UTTARGARAI BELT',
                'HESATHU-BELBATHAN BELT', 'HETRI BELT', 'HETRI PARALLEL BELT','HUTTI MASKI BELT', 'JAMDA-KOIRA BELT', 'JHAKRA PARSOLA BELT',
                'JHAKRA-PARSOLA BELT', 'JONAGIRI SCHIST BELT', 'JONNAGIRI SCHIST BELT', 'KALAJODA-GOLBADSHAPUR BELT',
                'KANJAMALLAI-ATTUR BELT', 'KHANKHERA BELT', 'KHETRI BELT','KHETRI PARALLEL BELT', 'KHETRI PARELLEL BELT', 'KOLAR BELT',
                'KUTCH BELT', 'MAHAKOSHAL BELT', 'MAILARAM BELT', 'MALANJKHAND BELT', 'MUNGER BELT', 'NELLORE BELT',
                'NOAMUNDI-JAMDA BELT', 'NOAMUNDI-JAMDA SECTOR', 'NORTH PURULIA SHEAR ZONE', 'NUGGIHALLI BELT', 'OLAR BELT',
                'PADAR-KI-PAL BELT', 'PALAMOU-RANCHI PROVINCE', 'PULIVENDLA BELT', 'PUR-BANERA BELT', 'PUR-BANERA BELT (EASTERN ZONE)',
                'PUR-BANERA BELT(EASTERN ZONE)', 'PUR-BANERA BELT(WESTERN ZONE)', 'RAJPURA-DARIBA-BETHUMNI BELT', 'RAJPURATDARIBA-BETHUMNI BELT',
                'RAMAGIRI BELT', 'RAMAGIRI-PENAKACHERLA-HUNGUD BELT', 'RAMPURA- GUCHA BELT', 'RANCH PLATEAU PROVINCE',
                'RANCHI PLATEAU PROVINCE', 'RANCHI-ADAR PARE PLATEAU', 'RORO-JOJOHATU BELT', 'ROWGHAT BELT',
                'SAKOLI FOLD BELT ( KOLARI-THAMBETHANI BELT)', 'SAKOLI FOLD BELT (AGARGAON-KHOBANA-KUHI BELT)',
                'SANDUR BELT-BELLARY SECTOR', 'SANDUR BELT-BTV RANGE', 'SANDUR SCHIST BELT-NEB RANGE',
                'SANDUR-KUMARSWAMI-DONIMALAI RANGE', 'SANDUR-UBBALAGUNDI RANGE', 'SARGIPALLI BELT', 'SARGUJA BELT', 'SAUSAR BELT', 'SAWAR BELT',
                'SHIMOGA BELT', 'SHIMOGA SCHIST BELT', 'SHIMOGA SCHIST BELT(KAPPALGUDDA RANGE)', 'SHIMOGA SCHIST BELT(RAMANDUR RANGE)', 'SHIMOGA-GOA BELT',
                'SINDUVALLI - TALLUR BELT', 'SINGBHUM BELT', 'VARIKUNTA-ZANGAMARAJUPALLE', 'WESTERN GHAT BELT',
                'WURRYA HILL RANGE', 'ZAWAR BELT'))
    LOCALITY = st.sidebar.selectbox('LOCALITY',('ADASH', 'AGARGAON', 'AJAR! - DANVA', 'AJITBURU MINES','AJJANAHALLI', 'AKOLA', 'AKWALI', 'ALADAHALLI', 'AMARKANTAK',
               'AMBALAVYAL', 'AMBAMATA', 'AMJHARA', 'AMPTIPANI', 'ANAMINI PARBAT','ANDERKOLWA', 'ANMOD', 'ARAKU VALLEY AREA', 'ATTIGUNDI,GALLIKERA',
               'ATTIKATTI BLOCK', 'ATTUR', 'BADAMGARH PAHAR', 'BAGHDAPA', 'BAHARAGORA', 'BAILADILA', 'BAIRAPUR', 'BALAGHAT', 'BALARIA',
               'BALDA', 'BALESWAR', 'BALIA PAHAR-KHEJURDAR/', 'BALLALAYAYANADURG HILLOCK', 'BANDALAMOTTU', 'BANERA R.F',
               'BANGROBAR', 'BANWAS', 'BARAGONDA', 'BARAIBURU', 'BARAJAMDA', 'BARKADIH-GHARGHUTA NADI', 'BAROI MOGRA', 'BASANTGARH',
               'BAULA-NAUHASI BLOCK', 'BEKU', 'BELANG I', 'BELBATHAN', 'BELEGAL,HARAGONDONA,TUM.TONNES1', 'BELGUMBA', 'BELLENAHALLI',
               'BETHUMNI', 'BETJHARAN', 'BHADRAMPALLE', 'BHAGLATOLI AREA', 'BHAGONI', 'BHAKTARAHALLI BLOCK', 'BHANDARBOLI', 'BHITARAMDA',
               'BHIYABASA', 'BHUKIA', 'BICHIWARA', 'BIRHNI-GARKHASWAR-AGARDIH', 'BISAI', 'BISANATTAM MINE PROSPECT', 'BOGRU HILL', 'BOPHLIMATTI',
               'BOYANBILL', 'C.K.HALLI', 'CHANDIPAT AREA', 'CHANDMARI', 'CHICHROLI', 'CHIGARGUNTA BLOCK', 'CHIKANAKAHALLI',
               'CHIKANYAKANAHALLI', 'CHIKKONAHALL1', 'CHINCHERGI', 'CHINMULGUND', 'CHIRIA', 'CHIRO', 'CHIROPATTOLI', 'CHITAR',
               'CHITTORI-TAINANDAMALAI', 'CODGUI', 'COLAMBA', 'DARIBA-RAJPURA', 'DAUKI-DUBA NALA', 'DEDWAS(NORTH)', 'DEDWAS(SOUTH)', 'DEGANA',
               'DERI', 'DEVADA', 'DEVPURA(RANNINGpURA)', 'DEVPURA-BANERA','DEWAKA BERA', 'DHADKIDIH', 'DHANOTA', 'DHANSUA-LAUGHAR-JAGANTOLI',
               'DHAULA', 'DHOLAMALA', 'DHUKONDA', 'DONA EAST BLOCK','DONA TEMPLE BLOCK', 'DUMERDIHA', 'DUMKONDA',
               'FULIHARI,KURMITAR,SILJORA', 'G.R.HALLI', 'GADARIAKHERA', 'GADERIAKHERA', 'GANDHAMADAN PLATEAU', 'GANI KALVA', 'GARBHAM',
               'GARDA', 'GHATKURI', 'GHATTIHOSAHALLI', 'GHUGRA-KAYAR','GHUGROTOLLI', 'GOLBADSHAPUR', 'GOLIA', 'GOLLAPALLE', 'GORIA',
               'GUDEM AREA', 'GUMGAON', 'GURHARPAHAR', 'GURLA', 'GURLIDU GROUP', 'HALKUNDI,HONNAHALI', 'HAMETAMOGRA', 'HARENABALLI', 'HATHORI',
               'HATTIKAMBA', 'HIRA BUDINI', 'HOSUR-YELISHIRUR', 'HULLIKATTE', 'HUTTI MINE BLOCK', 'IMALIYA', 'INGALDHALU', 'INGILIGI',
               'ITAR-BALIJODI', 'JAGPURA', 'JAISINGpUR (9 MINES)', 'JAMBUA', 'JAMBUR', 'JAYAPURA', 'JHAKRA', 'JODA', 'JUNEWANI', 'KABULIYATKATTI BLOCK', 'KADALAGUDDA', 'KADONI', 'KALABAR',
               'KALAPAHARI', 'KALASPURA', 'KALLAHALLI (3 MINES)', 'KALTA', 'KANTORIA', 'KAPPALGUDDA', 'KARADIKOLLA', 'KARAMPADA', 'KAREKUCHI',
               'KARINAGUR', 'KARLAPAT', 'KARNAPODIKONDA', 'KATHAKHAL-MANJIMLI', 'KATSAI', 'KEMMANAGUNDI', 'KEMPINKOTE MINE', 'KENDADIH',
               'KHANDBAND', 'KHANDELA', 'KHANKHERA', 'KHARGPUR HILLS', 'KHERJI-KHERA', 'KISANWALI', 'KODINGAMALI', 'KOIRA', 'KOLAR MINE',
               'KOLIHAN', 'KOTAPALLE BLOCK', 'KOTEBARA', 'KOTTAPALLE PROSPECT', 'KOYADONGRI', 'KUDERMUKH', 'KUDITHAINAPALLE', 'KUKUD',
               'KUMARDHUBI', 'KUMARSWAMI', 'KUMITRA', 'KUNCHIGANAHALU', 'KUNDERKOCHA(POROJHARNA MINE)', 'LAKEND', 'LAUGHAR-KAMHATOLA',
               'LOKIPAT', 'MADERAHALLI', 'MADHANKUDAN', 'MAILARAM', 'MAINAJHARIA','MALANGTOLI', 'MALANJKHAND', 'MALIKHERA',
               'MALIKHERA(EASTERN ZONE)', 'MALIPARBAT', 'MALLAPAKONDA', 'MALWALI', 'MANKAMACHA', 'MASANIKERE', 'MEDIKERIPURA', 'MERALGORA',
               'MIDIPENTA', 'MOCHIA', 'MOSABONI', 'MUNDVAL', 'MYSORE MINE BLOCK', 'NALLAKONDA', 'NANDUP', 'NAOGOWN', 'NARAYANPURA', 'NAUGAL',
               'NAVALUTTI', 'NETRA', 'NISHIKHAL', 'NITHOR', 'NOAMUNDI', 'NOTABURU', 'NUGGIHALLI', 'NUIA', 'OBULAPURAM', 'ORSAPAT',
               'PACHERI-MURADPUR', 'PADAR-KI-PAL', 'PADUNA', 'PANCHALA', 'PANCHPATMALI', 'PAPINAYAKANAHALLI', 'PARSOLA', 'PATHARGORA',
               'PENUKONDA PROSPECT', 'PIMPERGUDA-PIMPERKUNTA', 'PIMPERKUNTA', 'PIPELA', 'POTANGI AREA', 'PUR-DARIBA', 'PURNAPANI', 'RAIRANGPUR',
               'RAKHA', 'RAMANDURG RANGE', 'RAMAPURAM', 'RAMCHANDRA PAHAR',  'RAMPURA-AGUCHA', 'ROIDA', 'RORO,JOJOHATU', 'ROWGHAT',
               'SALADIPURA', 'SAMODI', 'SANBAL', 'SANDUR', 'SANGLI MINE','SANKALAPURAM', 'SARAMDAH-BHADRASAI', 'SARGIPALLI', 'SARKUNDA',
               'SASBOHUMALI-PASANGMALI', 'SATDHUDIA', 'SATKUI', 'SAWAR','SIDESHWAR', 'SIJIMALI', 'SIMANDUNA', 'SINDESWARKALAN',
               'SINDUVALLI', 'SINGHANA', 'SONARIA-RUPARIA', 'SULAIPET', 'SURAPALLE PROSPECT', 'SURDA', 'SURJAGARH', 'TAGADUR BLOCK',
               'TALLUR', 'TAMAPAHAR', 'TAMOLGARH', 'TEJWALA', 'TERALI-BISGOD', 'THAKURANI', 'THAMBETHANI', 'TIKHI', 'TIRANGA', 'TIRODI',
               'TUPPADHUR', 'TURAMDIH', 'UDWARIA', 'UKWA', 'UTI', 'UTI TEMPLE BLOCK', 'VELAMPATTI', 'VENKATAPURAM',
               'VENUMBAKA-RASANUR', 'VITTALPURAM', 'WANDALLI', 'WARI', 'YEPPAMANA GOLD MINE', 'YERAPPA-GENTALLAPPA BLOCK', 'YOGIMALAI',
               'ZAWAR MALA'))
    STATE = st.sidebar.selectbox('STATE',('ANDHRA PRADESH','CHATTISGARH','GOA','GUJARAT','JHARKHAND','KARNATAKA','KERALA','MADHYA PRADESH','MAHARASTRA','ORISSA','RAJASTHAN','TAMILNADU'))
    TOPOSHEET = st.sidebar.selectbox('TOPOSHEET',('41 E','44 L','44 P','45 D','45 H','45 J','45 K','45 L','45 M',
    '46 E','46 I','48 I','48 K','48 M','48 N','54','54 A','54 B','55 P','56','56 D','56 I',
    '56 P','57','57 A','57 A','57 B','57 B','57 C','57 D','57 E','57 J',
    '57 L','57 N', '58 A','58 I','63 L','64','64 B','64 B','64 C','64 C','64 I',
    '64 L','64 M', '64 N','65','65 A','65 C','65 E','65 G','65 I','65 J',
    '65 J','65 N', '65 M','65 N','72 H','72 L','72 P','73','73 A','73 C','73 E',
    '73 G','73 J', '73 K'))
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