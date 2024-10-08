# starbucks
Starbucks Capstone Challenge Udacity

## Summary
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offers during certain weeks. 

## Problem Statement
Build a machine learning model that predicts offers based on user information and offer event

## File Description
~~~~~~~
        starbucks
          |-- app
                |-- templates
                        |-- fe.html
                |-- run.py
          |-- data
                |-- transcript.json
                |-- portfolio.json
                |-- profile.json
                |-- portfolio_for_run.json
          |-- models
                |-- train_classifier.py
          |-- Starbucks_Capstone_notebook.ipynb
          |-- Starbucks_Capstone_notebook-zh.ipynb
          |-- README
~~~~~~~
## Installation
Run `pip install -r requirements.txt`
List library:
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scikit-learn
- plotly

## Instructions:
1. Run run ML pipeline that trains classifier and saves
        `python models/train_classifier.py models/classifier.pkl`
2. Go to `app` directory: `cd app`
3. Run your web app: `python run.py`
4. Go to http://127.0.0.1:5000/

## Licensing, Authors, Acknowledgements
Udacity

