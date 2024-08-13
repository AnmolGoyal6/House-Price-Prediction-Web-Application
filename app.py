from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and preprocessing objects
with open('house_price_model.pkl', 'rb') as f:
    pipe = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        location = request.form['location']
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])

        input_data = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_data)[0]

        return render_template('index.html', prediction_text=f'Predicted House Price: ₹ {prediction:.2f} Lakhs')

if __name__ == '__main__':
    app.run(debug=True)
