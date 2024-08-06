from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model (ensure you have a model.pkl file in the same directory)
with open('TCS_stock_prediction.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get the input values from the form
            high = float(request.form['high'])
            low = float(request.form['low'])
            open_price = float(request.form['open'])
            volume = float(request.form['volume'])
            
            # Prepare the input data for the model
            features = np.array([[high, low, open_price, volume]])
            
            # Predict the close price
            predicted_close = model.predict(features)[0]
            
            return render_template_string(template, predicted_close=predicted_close)
        except ValueError:
            return "Please enter valid numerical values."
        except Exception as e:
            return f"An error occurred: {e}"
    
    return render_template_string(template)

template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Predictor</title>
</head>
<body>
    <h1>Stock Price Predictor</h1>
    <form method="post">
        <label for="high">Previous Day High:</label>
        <input type="text" id="high" name="high" required><br><br>
        <label for="low">Previous Day Low:</label>
        <input type="text" id="low" name="low" required><br><br>
        <label for="open">Previous Day Open:</label>
        <input type="text" id="open" name="open" required><br><br>
        <label for="volume">Previous Day Volume:</label>
        <input type="text" id="volume" name="volume" required><br><br>
        <input type="submit" value="Predict Close Price">
    </form>
    
    {% if predicted_close is not none %}
    <h2>Predicted Close Price: {{ predicted_close }}</h2>
    {% endif %}
