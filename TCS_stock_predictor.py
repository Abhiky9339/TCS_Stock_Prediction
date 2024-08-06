from flask import Flask, render_template, request
import pickle
import logging

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
with open('TCS_stock_prediction.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features from the form
            open_price = float(request.form['open'])
            high_price = float(request.form['high'])
            low_price = float(request.form['low'])
            volume = float(request.form['volume'])

            # Prepare the input for the model
            features = [[open_price, high_price, low_price, volume]]
            
            # Make prediction
            prediction = model.predict(features)[0]

            logging.info(f"Prediction made with input: {features}, Result: {prediction}")
            
        except Exception as e:
            logging.error("Error during prediction", exc_info=True)
            prediction = "Error in prediction. Please check the input values."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
