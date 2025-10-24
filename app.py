

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)


# now  Loading  and preprocessing the dataset

df = pd.read_csv("housing.csv")

#   checking the null values using isnull()

# df.isnull().sum()


# Handling  missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Spliting  data  into x and y variables
X = df.drop('price', axis=1)
y = df['price']

# Scaling  and training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)


# 2️ step  Flask routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {
            'area': float(request.form['area']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'stories': int(request.form['stories']),
            'parking': int(request.form['parking']),
            'mainroad': int(request.form['mainroad']),
            'guestroom': int(request.form['guestroom']),
            'basement': int(request.form['basement']),
            'hotwaterheating': int(request.form['hotwaterheating']),
            'airconditioning': int(request.form['airconditioning'])
        }

        # Prepare input
        sample = [[
            form_data['area'], form_data['bedrooms'], form_data['bathrooms'],
            form_data['stories'], form_data['parking'], form_data['mainroad'],
            form_data['guestroom'], form_data['basement'],
            form_data['hotwaterheating'], form_data['airconditioning']
        ]]

        sample_scaled = scaler.transform(sample)
        predicted_price = model.predict(sample_scaled)[0]

       
        def yes_no(value): return "Yes" if value == 1 else "No"

        form_data_display = {
            "Area (sq. ft)": f"{form_data['area']}",
            "Bedrooms": form_data['bedrooms'],
            "Bathrooms": form_data['bathrooms'],
            "Stories": form_data['stories'],
            "Parking": form_data['parking'],
            "Main Road": yes_no(form_data['mainroad']),
            "Guestroom": yes_no(form_data['guestroom']),
            "Basement": yes_no(form_data['basement']),
            "Hot Water Heating": yes_no(form_data['hotwaterheating']),
            "Air Conditioning": yes_no(form_data['airconditioning'])
        }

        return render_template(
            'index.html',
            prediction_text=f"Predicted House Price: ₹{predicted_price:,.2f}",
            user_inputs=form_data_display
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)




