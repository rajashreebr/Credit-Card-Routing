import streamlit as st
import catboost
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from category_encoders import OneHotEncoder
import joblib
from catboost import CatBoostClassifier

# Load the LightGBM model (assuming the model is saved in the file 'lgb_model.txt')
# model = lgb.Booster(model_file='lgb_model.txt')

# Define the Streamlit app
def app():
    st.title('Prediction Model')

    # catboost ['PSP', 'card', 'isof_flag', 'amount', 'day', 'hour', 'day_of_week']
    # isof ["country", "amount", "PSP", "3D_secured", "card"]
    # Create input fields for the user to enter the data
    country = st.selectbox('Country', ['Germany', 'Switzerland', 'Austria'])
    card = st.selectbox('Card',["Master","Visa","Diners"])
    secured3D = st.selectbox('3D Secured',[0,1])
    amount = st.number_input('Amount', min_value=0.0, format='%f')
    date_input = st.date_input("Payment Date")
    time_input = st.time_input("Payment Time")
    day = date_input.day
    hour = time_input.hour
    day_of_week = date_input.weekday()

    PSPs = ["Moneycard","Goldcard","UK_Card","Simplecard"]

    # Load the models 
    oh_encoder = joblib.load('models/oh_encoding.pkl')
    isof_model = joblib.load('models/isof_model.pkl')
    catboost_model = CatBoostClassifier()
    catboost_model.load_model('models/best_model.cbm')

    # Button to make prediction
    if st.button('Predict'):
        # Predict for each PSP and choose the best PSP which has optimal fee
        # Convert to onehotencoding and predict with Isolation Forest then predict with catboost

        optimal_PSP =""
        optimal_fee = 0

        for PSP in PSPs:
            isof_input = pd.DataFrame({
                "country": [country],
                "amount": [amount],
                "PSP": [PSP],
                "3D_secured": [secured3D],
                "card": [card]
            })

            isof_input = oh_encoder.transform(isof_input)
            isof_prediction = isof_model.predict(isof_input)

            # Catboost prediction 
            catboost_input = pd.DataFrame({
                "PSP": [PSP],
                "card": [card],
                "isof_flag": isof_prediction,
                "amount": [amount],
                "day": [day],
                "hour": [hour],
                "day_of_week": [day_of_week]
            })

            catboost_prediction = catboost_model.predict(catboost_input)

            
            catboost_predict_proba = catboost_model.predict_proba(catboost_input)

            # Caluclate the fee based on the prediction probability

            success_prob = catboost_predict_proba[0][1]
            failure_prob = catboost_predict_proba[0][0]

            if PSP == "Moneycard":
                success_fee = 5
                failure_fee = 2
            elif PSP == "Goldcard":
                success_fee = 10
                failure_fee = 5
            elif PSP == "UK_Card":
                success_fee = 3
                failure_fee = 1
            elif PSP == "Simplecard":
                success_fee = 1
                failure_fee = 0.5

            # Calculate the fee
            fee = (success_prob * success_fee) + (failure_prob * failure_fee)
            st.write(f"Predicted Fee for {PSP} is {fee}")

            if fee > optimal_fee:
                optimal_fee = fee
                optimal_PSP = PSP
                success_prob = catboost_predict_proba[0][1]

        st.markdown(f"**Optimal Payment Service Provider (PSP):** {optimal_PSP}")
        st.markdown(f"**Associated Fee:** {optimal_fee} Euro")
        st.markdown(f"**Probability of Success:** {success_prob*100:.2f}%")


# Run the app
if __name__ == '__main__':
    app()
