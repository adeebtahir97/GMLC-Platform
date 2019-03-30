import pandas as pd
import numpy as np
import pickle
import time
import json
import asyncFunctionality
import requests
from sklearn.preprocessing import LabelEncoder
from elasticsearch import helpers, Elasticsearch
from pandas.io.json import json_normalize
from elasticsearch.helpers import scan as escan
from collections import deque
import json
import flask
from flask import jsonify, request, Flask, request
import os
import certifi
from config import api
from flask_cors import CORS

content = {}

# app = Flask("after_response")
app = Flask(__name__)
CORS(app)

@app.route('/autocoding/preprocess', methods=["POST"])
def preprocess():
    global content
    print('Prep Started!')
    content = request.get_json()
    print('Content is->',content)
    inputParams = str(content["inputParams"]).split(',')
    outputParams = str(content["outputParams"]).split(',')
    bucketParams = str(content["bucketParams"]).split(',')
    elasticIndex = str(content["sourceIndex"])
    destinationIndex=str(content["destinationIndex"])
    callbackUrl = content.get("callbackUrl","")

    print(f'Index:: {elasticIndex}')
    uid = str(content["uid"])

    uid_set = {"bucketParams": bucketParams,
               "inputParams": inputParams, "outputParams": outputParams}
    uid_dict = {}
    # Check if Uid file exists
    if os.path.isfile("./uidDict.pickle"):
        with open("./uidDict.pickle", "rb") as pickle_in:
            uid_dict = pickle.load(pickle_in)
        if uid in uid_dict:
            pass
        else:
            uid_dict.update({uid: uid_set})
            pickle_out = open("./uidDict.pickle", "wb")
            pickle.dump(uid_dict, pickle_out, -1)
            pickle_out.close()
    else:
        uid_dict = {uid: uid_set}
        pickle_out = open("./uidDict.pickle", "wb")
        pickle.dump(uid_dict, pickle_out, -1)
        pickle_out.close()

    nbParams = [item for item in inputParams if item not in bucketParams]


    df = pd.read_csv(elasticIndex,header=0)

    print('(Rows,Cols):', df.shape)
    print('Preprocessing Started')
    t1 = time.time()
    for col in list(df.columns.values):
        df[col] = df[col].astype(str)

    # Remove Special Characters from all non-bucket columns
    for col in nbParams:
        df[col] = df[col].str.replace(r'[^\w+\s]', ' ')
        df[col] = df[col].str.replace(r' +', ' ')

    # Strip all trailing and leading white spaces
    for col in list(df.columns.values):        
        df[col].str.strip()

    # Replace Empty String Cells with NaN
    for col in list(df.columns.values):
        df[col] = df[col].replace(r'', np.nan, regex=True)

    # Remove rows with NaN key inputs
    df = df[pd.notnull(df[bucketParams]).all(axis=1)]

    # Replace Empty Cells with NA
    for col in list(df.columns.values):
        df[col] = df[col].replace(np.nan, 'NA', regex=True)

    # Make everything Upper Case
    for col in list(df.columns.values):
        df[col] = df[col].str.upper()

    # Creating a new column called 'key' which is concatenation of bucket params(fb_id,supp_code). This is also the key for the dictionary we'll make later
    df['key'] = df[bucketParams].apply(
        lambda x: ''.join(x.values.tolist()), axis=1)
    df['key'].astype(str)

    # The next two lines reposition the 'key' column. By default it will be at the last when its made. Now i'm making it come after fbid and suppcode
    my_column = df.pop('key')
    df.insert(2, my_column.name, my_column)

    # Add a column for concatenating all the Non-Bucket inputs.
 
    df['concat'] = df[nbParams].apply(lambda x: '##'.join(x.values.tolist()), axis=1)
    df['concat'].astype(str)

    # Concatenate the outputs into a column called target
    df['target'] = df[outputParams].apply(lambda x: '##'.join(x.values.tolist()), axis=1)
    df['target'].astype(str)

    # Bucketing and LabelEncoding happen here
    print('Bucketing and Label Encoding has started')    
    unique_keys = list(df.key.unique())
    tttt = time.time()
    print('There are ', len(unique_keys), ' buckets')
    buckets = []
    for key in unique_keys:
        dfTemp = df.loc[df['key'] == key]
        dfTemp = dfTemp.astype(str)

        #Sort the bucket rows by frequency of target
        dfTemp['count'] = dfTemp['target'].map(dfTemp['target'].value_counts())
        dfTemp.sort_values('count',inplace=True,ascending=False)
        dfTemp.reset_index(drop=True)

        # Take the target column and make it into a list(we will gradually build this to contain the NonBucketParamConcats)
        listOfNonBucketParamConcats = dfTemp['concat'].tolist()

        bucketParamsPlusConcat = bucketParams + ['concat']

        #Categorical to Integer Encoding
        integerMapping = []        
        for ind,col in enumerate(bucketParamsPlusConcat):
            colList = dfTemp[col].tolist()
            mapping={b:a for a,b in enumerate(set(colList))}
            mapList = [mapping[k] for k in colList]
            dfTemp[col+'Label'] = mapList
            res = {}
            for num in range(len(colList)):
                res.update({colList[num]:mapList[num]})
            integerMapping.append(res)            
        buckets.append(
            [dfTemp, integerMapping, listOfNonBucketParamConcats, key])
    print('Bucketing and Label Encoding Done! Time Taken->',time.time()-tttt,' seconds')

    # PICKLE THE BUCKETS TO 'buckets.pickle'. This pickle file has the list of all buckets
    tttt = time.time()
    print('Serialization of Label Encoded Buckets Started...')
    pickle_out = open("./buckets/"+uid+"buckets.pickle", "wb")
    pickle.dump(buckets, pickle_out)
    pickle_out.close()
    print('Serialization of Label Encoded Buckets completed in ',time.time()-tttt,' seconds')
    print('Net Time Taken For Preprocessing->',time.time()-t1,' seconds')

    print('Creating Processed Data CSV->')
    df.to_csv(destinationIndex,header=True)
    print('All Done!')
    return '{"response":"Preprocessing Started"}'


if __name__ == '__main__':
    app.run(port=4040,debug=True)
