import time
import multiprocessing
import pickle
import pandas as pd
import numpy as np
import certifi
from sklearn.ensemble import RandomForestClassifier
import json
import requests
import asyncFunctionality
from elasticsearch import helpers, Elasticsearch
from pandas.io.json import json_normalize
from elasticsearch.helpers import scan as escan
from collections import deque
from flask import jsonify, Flask, request
from config import api
import os
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

content = 0
nest = 15
crit = 'entropy'
msl = 1
mf = 'auto'
md = None

def process(bucketList, model_dict, bucketParams):
    global nest
    global crit
    global md
    global msl
    global mf
    # print('This process has',len(bucketList),' buckets')
    for bucket in bucketList[:]:    
        df = bucket[0]
        integerMapping = bucket[1]
        listOfNonBucketParamConcats = bucket[2]
        key = bucket[3]
        print('Training for Key->', key)

        # These are the Label Encoded Column's ColumnNames
        labelEncodedColumns = [s+'Label' for s in bucketParams] + ['concatLabel']
        dfTemp = df[labelEncodedColumns]


        clf1 = RandomForestClassifier(n_estimators=nest, random_state=1,criterion=crit,max_depth=md, min_samples_leaf=msl, max_features=mf)
        clf1.fit(df[labelEncodedColumns], df['target'])
        # acc = 60
        # prfs = [1,1,1,1]

        nparray = dfTemp.to_numpy()
        targetnparray = df['target'].to_numpy()
        # print('numpy array is->\n',nparray,len(nparray),'\n\n\n\n')
        # print('target numpy array is->\n',targetnparray,len(nparray),'\n\n\n\n')

        X_train, X_test, y_train, y_test = train_test_split(nparray, targetnparray, test_size=0.33, random_state=1)
        
        y_pred = clf1.predict(X_test)        
        # print('y_testLength->',len(y_test),'y_predLength->',len(y_pred))


        acc = accuracy_score(y_test,y_pred)
        print('Acc->',acc)
        prfs = list(precision_recall_fscore_support(y_test, y_pred, average='weighted'))
        print('prfs->',prfs)

        # In this dictionary, each key will map to a list which has classifier(index 0),integerMapping(index 1),concats(index 2)
        model_dict.update({key: [clf1, integerMapping, listOfNonBucketParamConcats,acc,prfs]})


app = Flask("after_response")
CORS(app)
asyncFunctionality.AfterResponse(app)

@app.after_response
def training():    

    elasticIndex = str(content["sourceIndex"])
    uid = str(content["uid"])
    callbackUrl = content.get("callbackUrl","")

    print(f'index:{elasticIndex}')

    uidDict = {}
    pickle_in = open("./uidDict.pickle","rb")
    uidDict = pickle.load(pickle_in)
    pickle_in.close()
    bucketParams = uidDict[uid]["bucketParams"]

    # es = Elasticsearch([api["elastic"]["url"]], scheme=api["elastic"]["scheme"], ca_certs=certifi.where(), port=9200, timeout=300, max_retries=10, retry_on_timeout=True)

    # body = {
    #     "size": 10000,
    #     "query": {
    #         "match_all": {}
    #     }
    # }
    # response = escan(client=es,index=elasticIndex,query=body, request_timeout=300, size=10000)

    # # Initialize a double ended queue
    # output_all = deque()
    # # Extend deque with iterator
    # output_all.extend(response)
    # # Convert deque to DataFrame
    # df = json_normalize(output_all)
    # df = df[[x for x in df.columns if "_source." in x]]

    # df = df[['_source.'+s for s in bucketParams] +['_source.key']+ ['_source.target']]
    # df.columns = bucketParams + ['key'] + ['target']
    df = pd.read_csv(elasticIndex,header=0)
   
    # GET UNIQUE KEYS
    unique_keys = list(df.key.unique())

    pickle_in = open("./buckets/"+uid+"buckets.pickle", "rb")
    buckets = pickle.load(pickle_in)
    pickle_in.close()

    # THE FINAL DICTIONARY CONTAINING THE MODELS
    manager = multiprocessing.Manager()
    model_dict = manager.dict()

    # LIST OF PROCESSES
    process_list = []
    nProcesses = 0
    # Create a process for each key(hence bucket)
    divideBy = 50
    nbuckets = len(unique_keys[:])

    if nbuckets<140:
        divideBy = 5

    intDiv = nbuckets//divideBy
    floatDiv = nbuckets/divideBy

    if intDiv == floatDiv:
        nProcesses = intDiv
    else:
        nProcesses = intDiv+1

    for idx in range(1):
        startInd = idx*divideBy
        if (startInd+divideBy)<len(unique_keys):
            process_list.append(multiprocessing.Process(target=process, args=(buckets[startInd:startInd+divideBy], model_dict, bucketParams)))
        else:
            process_list.append(multiprocessing.Process(target=process, args=(buckets[startInd:len(unique_keys)], model_dict, bucketParams)))
            break

    t1 = time.time()
    print('Training Has Started')
    # process(buckets[0:49], model_dict, bucketParams)
    for p in process_list[:]:
        p.start()

    for d in process_list[:]:
        d.join()
    
    print('Total Training Time->', time.time()-t1, ' seconds')
    
    # PICKLE THE DICTIONARY OF MODELS CREATED TO 'uidtestModels.pickle'
    pickle_out = open("./testModels/"+uid+"testModels.pickle", "wb")
    pickle.dump(dict(model_dict), pickle_out,-1)
    pickle_out.close()    

    # if callbackUrl == "":
    #     kafkaPayload={
    #         "topic": uid,
    #         "payload": {
    #             "msg":"training-completed",
    #             "uid":uid,
    #             "jwt":content["jwt"]
    #         }
    #     }
    #     requests.post(url=api["kafkaPublish"]["url"], json=kafkaPayload, headers={'Content-Type': 'application/json','Authorization':api["kafkaPublish"]["authorization"]})
    # else:
    #     responseData = {
    #         "response":"Training completed!"
    #     }

    #     r1 = requests.post(url=callbackUrl, json=responseData, headers={'Content-Type': 'application/json'})
    #     print(r1.status_code, r1.reason, r1.text)
    print('Training completed & Models Dumped!')

@app.route('/autocoding/train', methods=["POST"])
def train():
    global content
    content = request.get_json()

    return '{"response":"Training Started!"}'

@app.route('/autocoding/model', methods=["POST"])
def model():    
    global nest
    global crit
    global md
    global msl
    global mf
    content = request.get_json()
    if len(content['nest']) > 0:
        nest = content['nest']
    if len(content['crit']) > 0:
        crit = content['crit']
    if len(content['md']) > 0:
        md = content['md']
    if len(content['msl']) > 0:
        msl = content['msl']
    if len(content['mf']) > 0:
        mf = content['mf']

    return '{"response":"Model Parameters Set!"}'

@app.route('/', methods=["GET"])
def hello():
    return '{"response":"Hello World!"}'

if __name__ == '__main__':
    app.run(port=4050,debug=True)


