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

# Binarize cases for logistic regression (above or below mean cases)
df_historical["high_case"] = (df_historical["cases"] > df_historical["cases"].mean()).astype(int)

# Splitting data
X = df_historical[["day"]]
y = df_historical["high_case"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict next day's case category
next_day = np.array([[31]])
predicted_class = model.predict(next_day)
predicted_label = "High Cases" if predicted_class[0] == 1 else "Low Cases"
print(f"Predicted case category for Day 31: {predicted_label}")

# Streamlit App
st.title("COVID-19 Cases Prediction in USA")
st.write("Predicting whether the next day's cases will be high or low based on historical data.")

day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction = model.predict([[day_input]])
    label = "High Cases" if prediction[0] == 1 else "Low Cases"
    st.write(f"Predicted category for day {day_input}: {label}")
