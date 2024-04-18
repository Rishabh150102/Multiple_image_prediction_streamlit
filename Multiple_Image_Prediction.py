import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import plotly.express as px
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive



st.set_page_config(layout="wide")


# Title
st.title("Field Health Analyzer")

st.subheader("Weather Details")

#Weather Data (MAIN)
with st.container(border=True):
  latitude = 26.0100
  longitude = 80.4221

  def fetch_weather_forecast():
      url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relativehumidity_2m,precipitation"
      response = requests.get(url)
      data = response.json()
      return (data['hourly']['time'], 
              data['hourly']['temperature_2m'], 
              data['hourly']['relativehumidity_2m'], 
              data['hourly']['precipitation'])

  # Fetch data
  time_data, temperature_data, humidity_data, precipitation_data = fetch_weather_forecast()

  # Create the charts
  fig_temp = px.line(x=time_data, y=temperature_data, title='Temperature Forecast')
  fig_humidity = px.line(x=time_data, y=humidity_data, title='Relative Humidity Forecast (%)')
  fig_rain = px.bar(x=time_data, y=precipitation_data, title='Precipitation Forecast (mm)')

  # Display in Streamlit (using tabs for better organization)
  tab1, tab2, tab3 = st.tabs(["Temperature", "Humidity", "Rain"])

  with tab1:
      st.plotly_chart(fig_temp)

  with tab2:
      st.plotly_chart(fig_humidity)

  with tab3:
      st.plotly_chart(fig_rain)

# Display Current Temprature (MAIN)   
      
def get_current_temperature():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"  
    response = requests.get(url)
    data = response.json()

    if 'current_weather' in data:
        return data['current_weather']['temperature']
    else:
        return None

with st.container(border=True):
   # Get the current temperature
  current_temperature = get_current_temperature()

  if current_temperature is not None:
      st.subheader("Current Temperature")
      st.metric("", f"{current_temperature}°C") 
      if current_temperature > 0:
         progress_bar_value = current_temperature / 60
         st.progress(progress_bar_value)
  else:
      st.warning("Unable to fetch current temperature data.")



# Code for Analysis of images starts (MAIN)


# Authentication
gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")  # Customizable path 
drive = GoogleDrive(gauth)

# Find your file 
file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
file = [f for f in file_list if f['title'] == 'resnet50_30epoch_multiclass.h5'][0] 

# Download the model
file.GetContentFile('resnet50_30epoch_multiclass.h5')  # Replaces with your file name

# Load the model
model = tf.keras.models.load_model('model.h5')

# model = tf.keras.models.load_model('C:\\Users\\risha\\Downloads\\resnet50_30epoch_multiclass.h5') 
# for github
# model = tf.keras.models.load_model('resnet50_30epoch_multiclass.h5') 


def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  
    return image_array

# Function to classify whether the field is healthy or unhealthy
# def classify_field(predictions):
#   avg_prediction = np.mean(predictions)
#   st.write(avg_prediction)
#   if avg_prediction >= 0.5:  
#     return "Healthy"
#   else:
#     return "Unhealthy"


st.subheader("Crop Health Analysis")


uploaded_files = st.file_uploader("Upload crop images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    healthy_count = 0
    predictions = []
    total_count = len(uploaded_files)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        
        image_array = preprocess_image(image, target_size=(224, 224)) 
        image_array = np.expand_dims(image_array, axis=0) 

        
        prediction = model.predict(image_array)
        st.write(prediction)
        predictions.append(prediction[0][0])

        #Display the image
        st.image(image)

        if prediction[0][0] >= 0.5:
            st.write('Healthy')
        else:
            st.write('Unhealthy')


    # Calculate percentages of healthy and unhealthy predictions
    healthy_count = sum(pred >= 0.5 for pred in predictions)
    unhealthy_count = len(predictions) - healthy_count
    total_count = len(predictions)
    healthy_percentage = (healthy_count / total_count) * 100
    unhealthy_percentage = (unhealthy_count / total_count) * 100

    # Classify whether the field is healthy or unhealthy based on stored predictions
    if(healthy_percentage > unhealthy_percentage):
       result = 'Healthy'
    else:
       result = 'Unhealthy'

    
    st.write("The field is:", result)
    st.write("Percentage of healthy predictions:", f"{healthy_percentage:.2f}%" )
    st.write("Percentage of unhealthy predictions:", f"{unhealthy_percentage:.2f}%" ) 

    # Pie Chart
    labels = ['Healthy', 'Unhealthy']
    values = [healthy_percentage, unhealthy_percentage]
    fig = px.pie(values=values, names=labels, title='Field Health Distribution')

    
    st.plotly_chart(fig)

    
