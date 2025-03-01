import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Fetch COVID-19 data
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
print(df)

# Visualizing COVID-19 Data
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
plt.show()

# Generate synthetic historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)
historical_deaths = np.random.randint(500, 2000, size=30)

# Creating dataset
df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Preparing data for logistic regression
df_historical["high_case"] = (df_historical["cases"] > df_historical["cases"].mean()).astype(int)

# Splitting data
X = df_historical[["day"]]
y = df_historical["high_case"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probability of high cases for the next day
next_day = np.array([[31]])
predicted_prob = model.predict_proba(next_day)[0][1]  # Probability of high cases
predicted_cases = int(predicted_prob * max(df_historical["cases"]))  # Scale to case count
print(f"Predicted cases for Day 31: {predicted_cases}")

# Streamlit App
st.title("COVID-19 Cases Prediction in USA")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction_prob = model.predict_proba([[day_input]])[0][1]
    predicted_cases = int(prediction_prob * max(df_historical["cases"]))
    st.write(f"Predicted cases for day {day_input}: {predicted_cases}")
