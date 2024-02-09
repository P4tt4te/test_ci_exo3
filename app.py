from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('./artifacts/regmodel.pkl')

@app.route("/")
def index():
    return render_template('home.html',prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_features = [np.array(features)]
    prediction = model.predict(input_features)

    return render_template('home.html', prediction_text=f"Predicted House Price: ${prediction[0]:.2f}")

if __name__=="__main__":
    app.run(debug=True)