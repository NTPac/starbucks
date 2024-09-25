import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import pickle

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

def create_model():
    if len(sys.argv) == 2:
        print('Clean data...')
        model_filepath = sys.argv[1:][0]
        for index, row in portfolio.iterrows():
            channels = row['channels']
            for channel in channels:
                if channel not in portfolio.columns:
                    portfolio[channel] = 0
                portfolio.loc[index, channel] = 1

        transcript.event = transcript.event.str.replace(' ', '_')
        transcript['offer_id'] = ""
        transcript['amount'] = 0.0
        for index, row in transcript.iterrows():
            if "offer_id" in row.value:
                transcript.loc[index, 'offer_id'] = row.value["offer_id"]
            if "offer id" in row.value:
                transcript.loc[index, 'offer_id'] = row.value["offer id"]
            if "amount" in row.value:
                transcript.loc[index, 'amount'] = row.value["amount"]
        transcript.drop('value', axis=1, inplace=True)

        merge_df = transcript.merge(profile, how='inner', left_on='person', right_on='id')
        merge_df.drop(columns=['id'], axis=1, inplace=True)
        merge_df = merge_df.merge(portfolio, how='left', left_on='offer_id', right_on='id')
        merge_df.drop(columns=['id'], axis=1, inplace=True)
        merge_df.loc[merge_df[merge_df.offer_id == ''].index, 'offer_id'] = np.nan
        offer_id = merge_df[ ~merge_df.offer_id.isna() ].offer_id.unique()
        offer_id = dict(zip(offer_id, list(range(0,len(offer_id)))))
        merge_df.offer_id = merge_df.offer_id.map(offer_id)
        merge_df.offer_id = pd.to_numeric(merge_df.offer_id)
        events = merge_df.event.unique()
        events = dict(zip(events, list(range(0,len(events)))))
        merge_df.event= merge_df.event.map(events)
        
        print('Building model...')
        df = merge_df.copy()
        df = df[df.offer_id.notna()]
        df.gender.fillna('O', inplace=True)
        genders = df.gender.unique()
        genders = dict(zip(genders, list(range(0,len(genders)))))
        df.gender = df.gender.map(genders)
        offer_types = df.offer_type.unique()
        offer_types = dict(zip(offer_types, list(range(0,len(offer_types)))))
        df.offer_type = df.offer_type.map(offer_types)
        df_obj= df.copy()
        list_offer_id = df_obj[df_obj.offer_id.notna()].offer_id.astype(int).sort_values().unique()
        df_obj[list_offer_id] = 0
        for index, row in df_obj.iterrows():
            try:
                df_obj.loc[index, row.offer_id] += 1
            except Exception as e:
                print(f"row error {index}, error {e}")
        df_obj=df_obj[df_obj.offer_id.notnull()]
        df_obj.fillna(0, inplace=True)
        df_obj.drop('channels', axis=1, inplace=True)
        dataset = df_obj
        dataset = dataset[['gender', 'age', 'income',  'became_member_on', 'event',  
         0,  1, 2,  3, 4, 5, 
           6, 7, 8,  9]].groupby(['gender', 'age', 
                                  'income',  'became_member_on',
                                    'event'], group_keys=True).sum().reset_index()
        dataset[[0,  1, 2,  3, 4, 5,  6, 7, 8,  9]] = dataset[[0,  1, 2,  3, 4, 5,  6, 7, 8,  9]] > 0
        dataset[[0,  1, 2,  3, 4, 5,  6, 7, 8,  9]] = dataset[[0,  1, 2,  3, 4, 5,  6, 7, 8,  9]].astype(int)
        X = dataset[['gender', 'age', 'income',  'became_member_on', 'event']]
        X = X.astype(int)
        Y = dataset[[0,                  1,
                            2,                  3,                  4,
                            5,                  6,                  7,
                            8,                  9]]
        Y = Y.astype(int)
        pipeline = Pipeline([
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        parameters = {
            'clf__estimator__n_estimators': [20, 50, 100],
            'clf__estimator__min_samples_split': [2],
        }

        cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2)

        cv.fit(X_train, y_train)
        best_params = cv.best_params_
        best_model = cv.best_estimator_
        
        print('Evaluating model...')
        y_pred = best_model.predict(X_test)
        cols = list(y_test.columns)
        for col in cols:
            index = cols.index(col)
            print(col, ':')
            print(classification_report(y_test[col], y_pred[:,index]))
            print('----------------------------------------------------------------------')
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        # Save the model as a pickle file
        with open(model_filepath, 'wb') as file:
            pickle.dump(best_model, file)
    else:
        print('Please provide the filepath of the pickle file to '\
              'save the model. \n\nExample: python '\
              'train_classifier.py classifier.pkl')

if __name__ == '__main__':
    create_model()
