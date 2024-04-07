"""
This is the main file for the streamlit app. It contains the code for the streamlit app.
https://tweetapp.streamlit.app
"""

# Importing the necessary libraries
import streamlit as st
import pickle

# User defined Class importing for cleaning the text
from main_function import X_clean

# Function to predict the tweet and display the result
def prediction(user_tweet, model):
    tweet_clean = X_clean(text = user_tweet, train = False)   # Create an instance of our class and pass tweet into it 
    user_tweet = tweet_clean.average_embed()   # Get the average embedding of the tweet
    predicted_output = model.predict([user_tweet])  # Predict the tweet using the trained model
    if predicted_output == 1:
        print (st.error('This tweet is about a disaster'))
    else:
        print(st.success("This tweet isn't about a disaster"))

# Setting the page config for the WebApp
st.set_page_config(page_title = "Disaster Detection App", page_icon = "Images/logo.png")

# Title of our WebApp
st.title('Disaster Detection App')

#  Sidebar for the WebApp, where the user can select the model which they want to use for Prediction
menu = st.sidebar.radio("Menu",["Home","XGBoost","Random Forest","Logistic Regression","KNN","Naive Bayes","Decision Tree","About"])

#  Conditional statements for the WebApp sidebar based on whether a new model or an existing model is selected 
if menu=="Home":
    st.image("Images/x.png",width=550)
    st.write("Welcome to the Disaster Detection App")
    st.write("This app is designed to predict whether a tweet is about a disaster or not.")
    st.write("Please select a model from the sidebar to get started.")

elif menu=="XGBoost":
    # Load the trained XGBoost model

    with open('trained_models/xgboost_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('XGBoost Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
elif menu=="Random Forest":
    # Load the Random Forest Classifier model
    with open('trained_models/randomforest_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Random Forest Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
elif menu=="Logistic Regression":
    # Load the Logistic Regression model
    with open('trained_models/logistic_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Logistic Regression Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
elif menu=="KNN":
    # Load the KNN model
    with open('trained_models/knn_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('KNN Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')     
elif menu=="Naive Bayes":
    # Load the Naive Bayes model
    with open('trained_models/nb_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Navie Bayes Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')

elif menu=="Decision Tree":
    # Load the Decision Tree model
    with open('trained_models/dt_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Decision Tree Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')

elif menu=="About":
    # Display the about us page
    with open('html_pages/About_us.html', 'rb') as f:
        aboutus = f.read()
        aboutus = aboutus.decode("utf-8")
    st.markdown(aboutus, unsafe_allow_html=True)
