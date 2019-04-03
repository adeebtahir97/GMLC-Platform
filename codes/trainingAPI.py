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
from pandas_ml import ConfusionMatrix
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

content = {}

def process(bucketList, model_dict, bucketParams,content):
    
    # print('This process has',len(bucketList),' buckets')
    nestimators = 15
    critt = 'entropy'
    maxd = 20 
    minsl = 1 
    maxf = 'auto'
    
    if len(content['nest']) > 0:
        nestimators = int(content['nest'])
    if len(content['crit']) > 0:
        critt = content['crit']
    if len(content['md']) > 0:
        maxd = int(content['md'])
    if len(content['msl']) > 0:
        minsl = int(content['msl'])
    if str(content['mf']) != "auto" and len(content['mf']) > 0:
        maxf = int(content['mf'])

    for bucket in bucketList[:]:    
        df = bucket[0]
        integerMapping = bucket[1]
        listOfNonBucketParamConcats = bucket[2]
        key = bucket[3]
        print('Training for Key->', key)

        # These are the Label Encoded Column's ColumnNames
        labelEncodedColumns = [s+'Label' for s in bucketParams] + ['concatLabel']
        dfTemp = df[labelEncodedColumns]

        print('Creating RF Tree with:')
        print('nest,crit,msl,mf,md->',nestimators,critt,minsl,maxf,maxd)
        clf1 = RandomForestClassifier(n_estimators=nestimators, random_state=1,criterion=critt,max_depth=maxd, min_samples_leaf=minsl, max_features=maxf)
        


        nparray = dfTemp.to_numpy()
        targetnparray = df['target'].to_numpy()
        # print('numpy array is->\n',nparray,len(nparray),'\n\n\n\n')
        # print('target numpy array is->\n',targetnparray,len(nparray),'\n\n\n\n')

        X_train, X_test, y_train, y_test = train_test_split(nparray, targetnparray, test_size=0.33, random_state=1)

        clf1.fit(X_train, y_train)
        y_pred = clf1.predict(X_test)        
        # print('y_testLength->',len(y_test),'y_predLength->',len(y_pred))


        acc = accuracy_score(y_test,y_pred)
        # cm = ConfusionMatrix(y_test,y_pred)
        # print(cm)
        print('Acc->',acc)
        prfs = list(precision_recall_fscore_support(y_test, y_pred, average='micro'))
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

    df = pd.read_csv('./datasets/'+elasticIndex,header=0)
   
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

    for idx in range(nProcesses):
        startInd = idx*divideBy
        if (startInd+divideBy)<len(unique_keys):
            process_list.append(multiprocessing.Process(target=process, args=(buckets[startInd:startInd+divideBy], model_dict, bucketParams,content)))
        else:
            process_list.append(multiprocessing.Process(target=process, args=(buckets[startInd:len(unique_keys)], model_dict, bucketParams,content)))
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


    print('Training completed & Models Dumped!')

@app.route('/autocoding/train', methods=["POST"])
def train():
    global content
    content = request.get_json()

    return '{"response":"Training Started!"}'

if __name__ == '__main__':
    app.run(port=4050,debug=True)


