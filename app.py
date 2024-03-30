import streamlit as st
import pickle
from main_function import X_clean

st.title('Disaster Detection App')

menu = st.sidebar.radio("Menu",["Home","XGBoost","Random Forest","Logistic Regression","KNN","Naive Bayes","Decision Tree","Neural Network","About"])

if menu=="Home":
    st.image("x.png",width=550)
    st.write("Welcome to the Disaster Detection App")
    st.write("This app is designed to predict whether a tweet is about a disaster or not.")
    st.write("Please select a model from the sidebar to get started.")

elif menu=="XGBoost":
    with open('xgboost_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    st.write('XGBoost Model')
    tweet = st.text_area('Enter Tweet')
    
    if st.button('Predict'):
        if tweet:
        #     prediction = loaded_model.predict([tweet])
            tweet_clean = X_clean(x = tweet)
            tweet = tweet_clean.average_embed()
            prediction = loaded_model.predict([tweet])
        #     prediction = 1
            if prediction == 1:
                st.write('This tweet is about a disaster')
            else:
                st.write('This tweet is not about a disaster')
        else:
            st.write('Please enter a tweet')

elif menu=="Random Forest":
    with open('randomforest_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    st.write('Random Forest Model')
    tweet = st.text_area('Enter Tweet')
    
    if st.button('Predict'):
        if tweet:
        #     prediction = loaded_model.predict([tweet])
            tweet_clean = X_clean(x = tweet)
            tweet = tweet_clean.average_embed()
            prediction = loaded_model.predict([tweet])
        #     prediction = 1
            if prediction == 1:
                st.write('This tweet is about a disaster')
            else:
                st.write('This tweet is not about a disaster')
        else:
            st.write('Please enter a tweet')
            
elif menu=="Logistic Regression":
    with open('logistic_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    st.write('Logistic Regression Model')
    tweet = st.text_area('Enter Tweet')
    
    if st.button('Predict'):
        if tweet:
        #     prediction = loaded_model.predict([tweet])
            tweet_clean = X_clean(x = tweet)
            tweet = tweet_clean.average_embed()
            prediction = loaded_model.predict([tweet])
        #     prediction = 1
            if prediction == 1:
                st.write('This tweet is about a disaster')
            else:
                st.write('This tweet is not about a disaster')
        else:
            st.write('Please enter a tweet')

elif menu=="KNN":
    with open('knn_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    st.write('KNN Model')
    tweet = st.text_area('Enter Tweet')
    
    if st.button('Predict'):
        if tweet:
        #     prediction = loaded_model.predict([tweet])
            tweet_clean = X_clean(x = tweet)
            tweet = tweet_clean.average_embed()
            prediction = loaded_model.predict([tweet])
        #     prediction = 1
            if prediction == 1:
                st.write('This tweet is about a disaster')
            else:
                st.write('This tweet is not about a disaster')
        else:
            st.write('Please enter a tweet')
            
elif menu=="Naive Bayes":
    with open('nb_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    st.write('Navie Bayes Model')
    tweet = st.text_area('Enter Tweet')
    
    if st.button('Predict'):
        if tweet:
        #     prediction = loaded_model.predict([tweet])
            tweet_clean = X_clean(x = tweet)
            tweet = tweet_clean.average_embed()
            prediction = loaded_model.predict([tweet])
        #     prediction = 1
            if prediction == 1:
                st.write('This tweet is about a disaster')
            else:
                st.write('This tweet is not about a disaster')
        else:
            st.write('Please enter a tweet')

elif menu=="Decision Tree":
    with open('dt_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    st.write('Decision Tree Model')
    tweet = st.text_area('Enter Tweet')
    
    if st.button('Predict'):
        if tweet:
        #     prediction = loaded_model.predict([tweet])
            tweet_clean = X_clean(x = tweet)
            tweet = tweet_clean.average_embed()
            prediction = loaded_model.predict([tweet])
        #     prediction = 1
            if prediction == 1:
                st.write('This tweet is about a disaster')
            else:
                st.write('This tweet is not about a disaster')
        else:
            st.write('Please enter a tweet')