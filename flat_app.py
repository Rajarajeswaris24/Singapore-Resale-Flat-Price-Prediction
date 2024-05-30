#import packages
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
import base64
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv(r"G:\guvi\resaleflat\singapore_flats2.csv")

#streamlit  background color
page_bg_color='''
<style>
[data-testid="stAppViewContainer"]{
        background-color:#FFDAB9;
}
</style>'''

#streamlit button color
button_style = """
    <style>
        .stButton>button {
            background-color: #ffa089 ; 
            color: black; 
        }
        .stButton>button:hover {
            background-color: #ffddca; 
        }
    </style>    
"""
#streamlit settings
st.set_page_config(
    page_title="Singapore Resale Flat Price Prediction",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="auto")


st.markdown(page_bg_color,unsafe_allow_html=True)  #calling background color
st.markdown(button_style, unsafe_allow_html=True)  #calling button color

st.title("Singapore Resale Flat Price Prediction")

#menu
selected = option_menu(menu_title=None,options= ["HOME", "PREDICT RESALE PRICE"],icons=["house", "cash-coin"],
          default_index=0,orientation='horizontal',
          styles={"container": { "background-color": "white", "size": "cover", "width": "100"},
            "icon": {"color": "brown", "font-size": "20px"},

            "nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2px", "--hover-color": "#ffe5b4"},
            "nav-link-selected": {"background-color": "#E2838A"}})


#loading saved model and encoder using pickle
file_encoder1 = r"G:\guvi\resaleflat\town_labelencoder.pkl"
with open (file_encoder1,'rb') as enc1:
    loaded_label_encoder1 = pickle.load(enc1)

file_encoder2 = r"G:\guvi\resaleflat\ftype_labelencoder.pkl"
with open (file_encoder2,'rb') as enc2:
    loaded_label_encoder2 = pickle.load(enc2)

file_encoder3 = r"G:\guvi\resaleflat\street_labelencoder.pkl"
with open (file_encoder3,'rb') as enc3:
    loaded_label_encoder3 = pickle.load(enc3)

file_encoder4 = r"G:\guvi\resaleflat\fmodel_labelencoder.pkl"
with open (file_encoder4,'rb') as enc4:
    loaded_label_encoder4 = pickle.load(enc4)

file_encoder5 = r"G:\guvi\resaleflat\block_labelencoder.pkl"
with open (file_encoder5,'rb') as enc5:
    loaded_label_encoder5 = pickle.load(enc5)

file_encoder6 = r"G:\guvi\resaleflat\storey_labelencoder.pkl"
with open (file_encoder6,'rb') as enc6:
    loaded_label_encoder6 = pickle.load(enc6)

file_ran_for_reg1 = r'G:\guvi\resaleflat\xgboostreg_hyp.pkl'
with open(file_ran_for_reg1, 'rb') as f1:
    loaded_reg_model = pickle.load(f1)

@st.cache_resource  

#function predict resale price
def predict_price(town,flat_type,street_name,flat_model,block,storey_range,floor_area_sqm,lease_commence_date,resale_year,resale_month):
    
    town_encoded = loaded_label_encoder1.transform([town])[0]
    flat_type_encoded = loaded_label_encoder2.transform([flat_type])[0]
    street_name_encoded = loaded_label_encoder3.transform([street_name])[0]
    flat_model_encoded = loaded_label_encoder4.transform([flat_model])[0]
    block_encoded = loaded_label_encoder5.transform([block])[0]
    storey_range_encoded = loaded_label_encoder6.transform([storey_range])[0]
    input_data = pd.DataFrame({
        'town': [town_encoded],
        'flat_type': [flat_type_encoded],
        'block': [block_encoded],
        'street_name': [street_name_encoded],
        'storey_range': [storey_range_encoded],
        'floor_area_sqm': float(floor_area_sqm),
        'flat_model': [flat_model_encoded],
        'lease_commence_date': float(lease_commence_date),
        'resale_year': int(resale_year),
        'resale_month': int(resale_month)
    })
    prediction = loaded_reg_model.predict(input_data)
    return prediction[0]

#function convert gif to base 64
def get_img_base64(file_path):
    with open(file_path, 'rb') as file:
        encoded = base64.b64encode(file.read()).decode()
    return encoded

#home page
if selected=="HOME":
    cola,colb=st.columns(2)
    with cola:
        st.image(r"G:\guvi\resaleflat\HDB-flats.jpeg")
    with colb:
        st.write("")
        st.write('''**The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat. There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration. A predictive model can help to overcome these challenges by providing users with an estimated resale price based on these factors.**''')
        st.write("")
        st.write('''**A machine learning model  that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.**''')
    st.header(f"Regression: :red[ XGBRegressor(Extreme Gradient Boosting)]")
    st.write('* ML Regression model which predicts continuous variable :violet[**‘Resale_Price’**].')
    st.write('- XGBoost, short for Extreme Gradient Boosting, is an optimized machine learning library for gradient boosting, a powerful ensemble technique used primarily for regression and classification tasks. It was developed to provide high performance, speed, and accuracy. The main idea behind gradient boosting is to build an ensemble of trees sequentially, where each subsequent tree corrects the errors of the previous trees.')


#predict reslae price page
if selected=="PREDICT RESALE PRICE":
    col1,col2=st.columns(2)
    with col1:
        town = st.selectbox("Choose Town", df['town'].unique(),key=1)
        flat_type = st.selectbox("Choose Flat Type", df['flat_type'].unique(),key=2)
        df_c=df[df['town']==town]
        street_name = st.selectbox("Choose Street name", df_c['street_name'].unique(),key=3)
        flat_model = st.selectbox("Choose Flat Model", df['flat_model'].unique(),key=4)
        storey_range = st.selectbox("Choose Storey Range", df['storey_range'].unique(),key=5)
        
    with col2:
        block = st.text_input("Enter Block No (eg:309,606D,202A,1G)")
        floor_area_sqm = st.text_input("Enter Floor Area sqm (Min:28 & Max:173)")
        lease_commence_date = st.text_input("Enter Lease commense year (Min:1966 & Max:2018)")
        resale_year = st.text_input("Enter Resale year (Min:1990, Max:2024)")
        resale_month = st.text_input("Enter Resale month (Min:1, Max:12)")
        
    
    if st.button("Predict price"):
        resale_price=predict_price(town,flat_type,street_name,flat_model,block,storey_range,floor_area_sqm,lease_commence_date,resale_year,resale_month)
        gif_path = r'G:\guvi\resaleflat\money.GIF'
        gif_base64 = get_img_base64(gif_path)
        html_content = f"""
        <div style="display: flex; align-items: center;">
            <h4 style="color: purple; margin-right: 10px;">Resale Price is $ {resale_price}</h4>
            <img src="data:image/gif;base64,{gif_base64}" width="150">
        </div>
        """
        
        st.markdown(html_content, unsafe_allow_html=True)  #calling gif near output