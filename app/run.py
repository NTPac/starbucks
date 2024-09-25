import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
import pandas as pd

import joblib# load model

model = joblib.load("../models/classifier.pkl")

app = Flask(__name__)

# Sample offers based on user information
offers = {}
with open('../data/portfolio_for_run.json', 'r') as file:
    json_temp = json.load(file)
    for item in json_temp:
        offers[item["id"]] = item
print(offers)
list_cols = [0,	1,	2,	3,	4,	5,	6	,7	,8,	9]
offers_encode = {
    0: '9b98b8c7a33c4b65b9aebfe6a799e6d9',
    1: '0b1e1539f2cc45b7b9fa7c272da2e1d7',
    2: '2906b810c7d4411798c6938adc9daaa5',
    3: 'fafdcd668e3743c1bb461111dcafc2a4',
    4: '4d5c57ea9a6940dd891ad53e9dbe8da0',
    5: 'f19421c1d4aa40978ebb69ca19b0e20d',
    6: '2298d6c36e964ae4a3e7e9706d1fb8c2',
    7: '3f207df678b143eea3cee63160fa8bed',
    8: 'ae264e3637204a6fb9bb56bc8210ddfd',
    9: '5a8bc65990b245e5a138643cd4eb9837'
}
gender_encode = {'F': 0, 'M': 1, 'O': 2}
def determine_offers(user_info):
    # Example logic to determine offers based on user information
    age = user_info.get('age', 0)
    income = user_info.get('income', 0)
    became_member_on = user_info.get('became_member_on', '2018-07-30')
    gender = user_info.get('gender', 'O')
    event = user_info.get('event', '')
    
    print(f"user_info: {user_info}")
    age =  "0" if age == "" else age
    income =  "0" if income == "" else income
    became_member_on = "2018-07-30" if became_member_on == "" else became_member_on

    print("age:", age)
    print("income:", income)
    print("became_member_on:", became_member_on)
    print("gender:", gender)
    print("event:", event)

    became_member_on = int(became_member_on.replace('-',''))
    if became_member_on > 20180730:
        became_member_on = 20180730
    # set data 
    data = [[gender_encode[gender], age, income, became_member_on, event]] 

    test_ = pd.DataFrame(data, columns=['gender', 'age', 'income',  'became_member_on', 'event'])
    predictions = model.predict(test_)
    # Simple logic to determine offers
    print(predictions)
    test_pred = pd.DataFrame(predictions, columns=list_cols)
    columns_with_values_gt_0 = [col for col in test_pred.columns if (test_pred[col] > 0).any()]
    return [offers.get(offers_encode[key], None) for key in columns_with_values_gt_0]


@app.route('/')
def index():
    return render_template(
        'fe.html'
    )
    # return send_from_directory('.', 'index.html')

@app.route('/get_offers', methods=['POST'])
def get_offers():
    user_info = request.json
    user_offers = determine_offers(user_info)
    return jsonify({'offers': user_offers})

# @app.route('/go')
# def go():
#     # # save user input in query
#     # query = request.args.get('query', '') 

#     # # use model to predict classification for query
#     # classification_labels = model.predict([query])[0]
#     # classification_results = dict(zip(df.columns[4:], classification_labels))

#     # This will render the go.html Please see that file. 
#     return render_template(
#         'fe.html'
#     )

if __name__ == '__main__':
    app.run(debug=True)
