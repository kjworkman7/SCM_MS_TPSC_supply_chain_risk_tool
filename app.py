import streamlit as st
import pandas as pd
import requests
from transformers import pipeline
sentiment_model = pipeline("sentiment-analysis")

st.title("📊 Supply Chain Risk Scorecard")

# Load data
df = pd.read_csv("sites.csv")

st.subheader("Site Data")
st.dataframe(df)

@st.cache_data(ttl=600)
def get_weather_risk(location):
    api_key = "c11bd0606824e96c587fea7426120806"
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    
    try:
        response = requests.get(url).json()
        weather = response["weather"][0]["main"]
        
        if weather in ["Thunderstorm", "Extreme"]:
            return 80
        elif weather in ["Rain", "Snow"]:
            return 50
        else:
            return 10
    except:
        return 20
@st.cache_data(ttl=600)
def get_news_data():
    api_key = "b9b120b93dfc4916aa9e151984bae3e5"
    
    url = f"https://newsapi.org/v2/everything?q=supply%20chain%20OR%20logistics&language=en&sortBy=publishedAt&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Debug print (optional)
        # st.write(data)

        if data.get("status") != "ok":
            return [], 20
        
        articles = data.get("articles", [])[:5]
        
        score = 0
        
        for article in articles:
            headline = article.get("title", "")
            
            if headline:
                result = sentiment_model(headline)[0]
                
                if result['label'] == 'NEGATIVE':
                    score += 20
                else:
                    score += 5
        
        return articles, score
    
    except Exception as e:
        st.write("Error loading news:", e)
        return [], 20

# --- Risk Calculation ---
def calculate_risk(row, news_risk):
    risk = 0

    risk += row["criticality_score"] * 0.4

    if row["capability_score"] < 50:
        risk += 40
    elif row["capability_score"] < 70:
        risk += 25
    else:
        risk += 10

    weather_risk = get_weather_risk(row["location"])
    risk += weather_risk * 0.2
    risk += news_risk * 0.2

    return round(risk, 2)

articles, news_risk = get_news_data()

df["Risk Score"] = df.apply(lambda row: calculate_risk(row, news_risk), axis=1)

# --- Dashboard Metrics ---
st.subheader("📊 Risk Overview")

col1, col2, col3 = st.columns(3)

avg_risk = df["Risk Score"].mean()
high_risk_count = len(df[df["Risk Score"] > 60])
max_risk_site = df.loc[df["Risk Score"].idxmax()]["site"]

col1.metric("Average Risk", round(avg_risk, 1))
col2.metric("High Risk Sites", high_risk_count)
col3.metric("Highest Risk Site", max_risk_site)

# --- Color-coded table ---
def color_risk(val):
    if val > 60:
        return 'background-color: red'
    elif val > 40:
        return 'background-color: orange'
    else:
        return 'background-color: green'

import numpy as np

df["risk_level"] = np.where(df["Risk Score"] > 70, "High",
                   np.where(df["Risk Score"] > 40, "Medium", "Low"))

st.dataframe(df)

st.subheader("📋 Risk Table")

st.dataframe(df.style.applymap(color_risk, subset=["Risk Score"]))

st.subheader("📍 Supply Chain Risk Map")

map_data = df.rename(columns={
    "latitude": "lat",
    "longitude": "lon"
})

st.map(map_data, size="Risk Score")

st.subheader("📰 Live Supply Chain News")

articles, _ = get_news_data()

if articles:
    for article in articles:
        st.write(f"• {article['title']}")
else:
    st.write("No news available.")

# AI-style insights
st.subheader("🤖 AI Risk Insights")

for index, row in df.iterrows():
    drivers = []

    # Capability driver
    if row["capability_score"] < 50:
        drivers.append("low operational capability")
    elif row["capability_score"] < 70:
        drivers.append("moderate capability constraints")

    # Criticality driver
    if row["criticality_score"] > 80:
        drivers.append("high site criticality")

    # External risk driver
    if news_risk > 40:
        drivers.append("negative global supply chain sentiment")

    # Fallback if no drivers
    if not drivers:
        drivers.append("stable operating conditions")

    # Risk messaging
    if row["Risk Score"] > 60:
        st.write(f"🔴 {row['site']} is HIGH RISK due to {', '.join(drivers)}.")
    elif row["Risk Score"] > 40:
        st.write(f"🟠 {row['site']} has MODERATE RISK driven by {', '.join(drivers)}.")
    else:
        st.write(f"🟢 {row['site']} is LOW RISK with {', '.join(drivers)}.")