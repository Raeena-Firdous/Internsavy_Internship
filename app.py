# Core Packages
import streamlit as st
from PIL import Image

# Data Viz Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# EDA Packages
import pandas as pd
import numpy as np

st.set_page_config(page_title='GAD Analysis',page_icon = 'logo.png', layout = 'wide', initial_sidebar_state = 'auto')
sns.set(rc={'figure.figsize':(20,15)})

DATA_URL = ('Admission_Predict_Ver1.1.csv')

st.markdown('# Graduate Admission Dataset')
st.markdown('### **Analysis of Graduate Admission Dataset**')

st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwcICAcHBwcHBwcHBwoHBwcHBw8ICQcKFREWFhURHxMYHSggGCYlJx8TITEhJSkrLi4uFx8zODMsNygtLisBCgoKDQ0NDg0NDjcZFRkrNysrLS03Ky03LSstLS03KysrKystKysrKystKysrKysrKysrKysrKysrKysrKysrK//AABEIAUsAmAMBIgACEQEDEQH/xAAZAAEBAQEBAQAAAAAAAAAAAAABAAIEBQb/xAAWEAEBAQAAAAAAAAAAAAAAAAAAARH/xAAYAQEBAQEBAAAAAAAAAAAAAAABAAIDBP/EABYRAQEBAAAAAAAAAAAAAAAAAAABEf/aAAwDAQACEQMRAD8A+8SLo5IgomERqBqKNCENEiENQxIwGIowNEwGBqGGAxEpIF5pELu8MTQhiahhBDRhihBRiIaiMRDSMRgKaBgaJgagKSSaeZCond4YTFDAYYYIYGjDEYGiYDA1CYDASRCGiRCiTA0GkkgXmQgx6HhhhgjUDRMEajLRMEMBhMDUDaIhBMKSaMKiBMKhBSKReUYi7vFDGmY0GjCI0CYRCGoYRCGjCIQ1DDAYiSCCYYI0CkiC8qEQx6HjjUMEaBMMBgaMKijLUaiSDTSiMRJgMBRUURMaZjQJSiBeXDBDHoeNowGMmFqMtQNEwQwNQmAhqEhIxoxloNGKApGNMtAmJRAvLajLUd3jJghDRagIaiaZagaJZIJhgMRMIMBJEKJaZMBaiEILy41BDHd4zDAYK1GiIYy0YYDA0iDESQkWjGYYi0WSCYYCC0gQnmmCNR2eUmAwGGGAwNQmAwGIhRNNIFIkJNNQxmEFqGMlJqIIFwGAurymGJQEmAwNFRIEpJEkJEkJFoslFqFkgtIJJwxoQx0eYmAwFFRBopJEoEFEJFpApFAyokslFpABOWGCF0ecqJAkgoohAlJIlAoohJNJkokhBNIJFzQhNuBISJLJCKBROoLUSRqSKBRKCRJCCaQSLn1aydbcGkzp1ElnSkdOsoFpM6dSJ1lItIakmkyUWkzpRKQBc5Z1a24NJnTqROs6tRaOsrUta1M6dWLWtWsoFos6kWiykmizpROoIFzpnVrbg0hq1I6WdRLWnWUE0mdOpE6yUda1MnQdJ1nSjrSZKOtaggXNq1lOjg1q0JInWUE1pZ1ai0hqSaWhJNILQWloSLRZOoksoFzIJ0cSWSEUEk0gtSJCRa1MlJpDSCTrKRaQ0hHUEi5kk25IhJEslEoEIoJJpApEslElkgkhItBIJzoQtuaSSSSKSSSJSiCRBSJCRJCSJCBMISL/2Q==");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

img = Image.open('toefl.jpg')
new_image = img.resize((1000, 500))
st.image(new_image,caption = 'Graduate Admission Dataset')

st.markdown('### **About the Dataset:**')
st.info('This dataset was built \
    with the purpose of helping students in \
        shortlisting universities with their profiles. \
            The predicted output gives them a fair \
                idea about their chances for a particular university. \
                    This dataset is inspired by the UCLA Graduate Dataset from Kaggle. \
                        The graduate studies dataset is a dataset which describes the probability of \
                            selections for Indian students dependent on the following parameters below.')

st.markdown('### **Dataset Info:**')
st.markdown('##### **Attributes of the Dataset:**')
st.info('\t 1. GRE Score (out of 340), \
        \n\t 2. TOEFL Score (out of 120), \
        \n\t 3. University Rating (out of 5), \
        \n\t 4. Statement of Purpose/ SOP (out of 5), \
        \n\t 5. Letter of Recommendation/ LOR (out of 5), \
        \n\t 6. Research Experience (either 0 or 1), \
        \n\t 7. CGPA (out of 10), \
        \n\t 8. Chance of Admittance (ranging from 0 to 1)') 

def load_data(nrows):
    df = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    df.set_index('Serial No.', inplace=True)
    df.rename(lowercase, axis='columns', inplace=True)
    return df

st.title('Lets explore the Graduate Admission Dataset')
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading graduate admissions dataset...')
# Load 500 rows of data into the dataframe.
df = load_data(500)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading graduate admissions dataset...Completed!')

# Explore Dataset
st.header('Quick  Explore')
st.sidebar.subheader('Quick  Explore')
st.markdown("Tick the box on the side panel to explore the dataset.")

st.markdown("""
<style>
    [data-testid=stSidebar] {
        color:black;
        background-image: url("https://media.istockphoto.com/id/1135638647/photo/white-lines-and-spheres.jpg?b=1&s=170667a&w=0&k=20&c=KlFMDLyLj_V_AbGjSXaUxcuIb_7NN1s8cBRPYlfZqhw=");
    }
</style>
""", unsafe_allow_html=True)

if st.sidebar.checkbox("Show Raw Data"):
    st.subheader('Raw data')
    st.write(df)
if st.sidebar.checkbox("Show Columns"):
    st.subheader('Show Columns List')
    all_columns = df.columns.to_list()
    st.write(all_columns)
if st.sidebar.checkbox('Statistical Description'):
    st.subheader('Statistical Data Descripition')
    st.write(df.describe())
if st.sidebar.checkbox('Missing Values?'):
    st.subheader('Missing values')
    st.write(df.isnull().sum())

st.header('Create Own Visualization')
st.markdown("Tick the box on the side panel to create your own Visualization.")
st.sidebar.subheader('Create Own Visualization')
if st.sidebar.checkbox('Count Plot'):
    st.subheader('Count Plot')
    column_count_plot = st.sidebar.selectbox("Choose a column to plot count.", df.columns[:5])
    fig = sns.countplot(x=column_count_plot,data=df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
if st.sidebar.checkbox('Distribution Plot'):
    st.subheader('Distribution Plot')
    column_dist_plot = st.sidebar.selectbox('Choose a column to plot density.', df.columns[:5])
    fig = sns.distplot(df[column_dist_plot])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Showing the Prediction Model
st.header('Building Prediction Model')
st.sidebar.subheader('Prediction Model')
st.markdown("Tick the box on the side panel to run Prediction Model.")

import pickle

if st.sidebar.checkbox('View Prediction Model'):
    st.subheader('Prediction Model')
    pickle_in = open('randomclassifier.pkl', 'rb')
    model = pickle.load(pickle_in)

    @st.cache()

    def prediction(gre, toefl, univ,sop, lor, cgpa, resc):
        
        if resc == 'Yes':
            resc = 1
        else:
            resc = 0
        
        prediction = model.predict([[gre, toefl, univ, sop, lor, cgpa, resc]])

        return prediction
    
    def main():       
        
        gre = st.slider("GRE Score (out of 340):", 0, 340, 0, step = 1)
        toefl = st.slider("TOEFL Score (out of 120):", 0, 120, 0, step = 1)
        univ = st.slider("University Rating Score (out of 5):", value = 0, min_value = 0, max_value = 5, step = 1)
        sop = st.slider("SOP Score (out of 5):", value = 0.0, min_value = 0.0, max_value = 5.0, step = 0.5)
        lor = st.slider("LOR Score (out to 5):", value = 0.0, min_value = 0.0, max_value = 5.0, step = 0.5)
        resc = st.selectbox('Research Experience:', ("Yes", "No"))
        cgpa = st.number_input('Enter CGPA (out of 10):')
        
        if st.button("Predict"): 
            result = prediction(gre, toefl, univ, sop, lor, cgpa, resc)

            if result==1:
                st.success("Congratulations! You are eligible to apply for the university!")
                st.balloons()

            else:
                st.success("Better Luck Next Time :)")
        
    if __name__=='__main__': 
        main()

st.sidebar.subheader('Data Source')
st.sidebar.info("https://www.kaggle.com/graduate-admissions")