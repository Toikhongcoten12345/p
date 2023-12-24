from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1) 
    loaded_model = pickle.load(open("random_forest_model.pkl", "rb"))  
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_list.values())))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'Death due to heart disease'
        else:
            prediction = 'No deaths due to heart disease'
        return render_template("result.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)