import streamlit as st
import pickle
import pandas as pd

# Define the list of teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# Load the trained model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set up the Streamlit app title
st.title('IPL Win Predictor')

# Widgets for selecting batting and bowling teams
batting_team = st.selectbox('Select the batting team', sorted(teams))
bowling_team = st.selectbox('Select the bowling team', sorted(teams))
selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target')
score = st.number_input('Score')
overs = st.number_input('Overs completed')
wickets = st.number_input('Wickets out')

# Prediction button
if st.button('Predict Probability'):
    # Calculate remaining runs, balls, wickets, current run rate, and required run rate
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Create input DataFrame for prediction
    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    # Predict the win and loss probabilities
    result = pipe.predict_proba(input_df)
    win_probability = result[0][1] * 100
    loss_probability = result[0][0] * 100

    # Display the results
    st.header(f"{batting_team} - Win Probability: {win_probability:.2f}%")
    st.header(f"{bowling_team} - Loss Probability: {loss_probability:.2f}%")
