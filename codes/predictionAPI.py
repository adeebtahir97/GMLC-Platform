import pickle
import json
from flask import Flask,request
import re
import os
import time
import pathlib
from collections import OrderedDict
import threading
from flask_cors import CORS
import random
modelDict ={}
uidDict = {}

def checkRemark(variableDictPartial,nbParams,remark):
	rlist = remark.split('##')
	# print('Remark->',rlist)
	for splitInd,param in enumerate(nbParams):		
		if variableDictPartial[param].upper() != "":
			if variableDictPartial[param].upper() not in rlist[splitInd]: 
				# print('False')
				return False
		else: continue
	# print('True')
	return True

def get_nbConcatsConsidered(nbParams,variableDictPartial,listOfNonBucketParamConcats,NonBucketParamConcatsConsidered):
	allEmpty = 1
	alreadySeen = {}
	for param in nbParams:
		if len(variableDictPartial[param])==0:
			variableDictPartial.update({param:""})
		else: 
			allEmpty=0

	# print('PartialVarsDict->',variableDictPartial)
	if allEmpty: 
		NonBucketParamConcatsConsidered = listOfNonBucketParamConcats
	else:
		NonBucketParamConcatsConsidered = [s for s in listOfNonBucketParamConcats if checkRemark(variableDictPartial,nbParams,s)]
	return NonBucketParamConcatsConsidered

app=Flask(__name__)
CORS(app)

def loadModels(uid=None):
	global modelDict, uidDict

	uidPath = os.path.abspath("./uidDict.pickle")

	print(f'Found uidDict file at {uidPath} , isAvailable = {os.path.isfile(uidPath)}, len(modelDict) is {len(modelDict)}')

	if len(modelDict)==0 and os.path.isfile(uidPath): #Means the server is restarted hence models aren't loaded yet
		print(f'No models loaded yet.. Autoloading available models..')
		pickle_in = open(uidPath,"rb")
		uidDict = pickle.load(pickle_in)
		pickle_in.close()
		# os.chdir("../testModels")
		for file in os.listdir("./testModels"):
			if file.endswith(".pickle"): #Bulk Load All the pickled model files into memory
				if os.path.getsize("./testModels/"+file) > 0: 
					pickle_in = open("./testModels/"+file,"rb")
					curUid = file.split("testModels.pickle")[0]
					print(f'Loading model with uid {curUid}')
					modelDict[curUid] = pickle.load(pickle_in)
					pickle_in.close()

	elif uid in modelDict: pass
	else:
		if uid is None:
			print(f'None received for uid - ignoring..')
			return
		#Load the Model from Pickle file
		abspath = pathlib.Path("./testModels/"+uid+"testModels.pickle").absolute()
		print(f'Loading pickle file from {str(abspath)}')
		pickle_in = open(str(abspath),"rb")
		modelDict[uid] = pickle.load(pickle_in)
		pickle_in.close()
	return

@app.route('/autocoding/predict/load',methods=["POST"])
def load():

	content = request.get_json()
	uid = content["uid"]

	loadModels(uid=uid)

	return '{"response":"loaded"}'

@app.route('/autocoding/predict',methods=["POST"])
	
def predict():
	global modelDict, uidDict
	#Data Needed For Prediction(Obtained From HTML form page)
	content = request.get_json()	

	tTemp= time.time()
	#get the inputParams,bucketParams and outputParams from the uidDict pickle file
	uid = content["uid"]
	
	inputParams = uidDict[uid]["inputParams"]
	bucketParams = uidDict[uid]["bucketParams"]
	nbParams = [item for item in inputParams if item not in bucketParams]	
	theInputs = content['predInputs'].split(',')
	bucketInputs = content['bucketInputs'].split(',')
	variableListDict = {}
	i=0
	for param in bucketParams:
		variableListDict[param+'List'] = [bucketInputs[i]]
		i = i+1
	i = 0
	for param in nbParams:
		variableListDict[param+'List'] = [theInputs[i]]
		i = i+1

	print('variableListDict->',variableListDict)
	predictionsReplied = []
	variableDict = {}

	print('Time for Initialization->{:f}'.format(round(time.time()-tTemp,10)),' seconds')

	t1=time.time()
	# bulkCount = len(content[inputParams[0]])
	bulkCount = 1
	for numInd in range(bulkCount):
		tTemp = time.time()		
		for param in inputParams:
			variableDict[param] = variableListDict[param+'List'][numInd].upper()
			if param not in bucketParams:
				variableDict[param] = re.sub('[^\w+\s]', ' ', variableDict[param]).strip()
				variableDict[param] = re.sub(' +', ' ', variableDict[param])						

		print('Time to set Variables->{:f}'.format(round(time.time()-tTemp,10)),' seconds')
		#Create a list to hold all the ranked predictions	
		overall_predictions=[]
		overall_predictions_dict = {}

		print('VariableDict->',variableDict)
		#Form the key by concatinating bucketParams
		key = ""
		for param in bucketParams:
			key+= variableDict[param]
		key = str(key.upper())
		

		if key not in modelDict[uid]:
			print('key not in dict')
			responseData = {"predictions":[]}
			return(json.dumps(responseData))

		
		tTemp = time.time()
		#Get the classifier,integerMapping and listOfNonBucketParamConcats	
		clf = modelDict[uid].get(key)[0]
		integerMapping = modelDict[uid].get(key)[1]
		listOfNonBucketParamConcats = modelDict[uid].get(key)[2]
		acc = modelDict[uid].get(key)[3]

		precision = modelDict[uid].get(key)[4][0]

		recall = modelDict[uid].get(key)[4][1]

		fscore = 2*(precision*recall)/(precision + recall)


		print('Time To Fetch classifier,integerMapping and listofnbpc->{:f}'.format(round(time.time()-tTemp,10)),' seconds')

		#Create a list to hold only the NonBucketParamConcats that contain the partialInputs
		NonBucketParamConcatsConsidered = []

		#Variables that store the partialInputs
		variableDictPartial = {}
	
		tTemp = time.time()
		for param in nbParams:
			variableDictPartial[param] = variableDict[param]
		print('Time to make PartialVars->{:f}'.format(round(time.time()-tTemp,10)),' seconds')

		tTemp = time.time()
		NonBucketParamConcatsConsidered=get_nbConcatsConsidered(nbParams,variableDictPartial,listOfNonBucketParamConcats,NonBucketParamConcatsConsidered)
		print('Time To Fetch Considered Records->{:f}'.format(round(time.time()-tTemp,10)),' seconds')


		if len(NonBucketParamConcatsConsidered)==0:
			responseData = {"predictions":[]}
			return(json.dumps(responseData))

		tNEW = time.time()
		if len(NonBucketParamConcatsConsidered) > 30:
			NonBucketParamConcatsConsidered = NonBucketParamConcatsConsidered[:30]
		for rindex,remark in enumerate(NonBucketParamConcatsConsidered[:]):
			
			if remark not in integerMapping[len(bucketParams)]:
				print(f'Skipping {remark} - not able to find a label encoded version')
				continue

			xtest = [integerMapping[ind][str(variableDict[param])] for ind,param in enumerate(bucketParams)]
			xtest+=[integerMapping[len(bucketParams)][remark]]
			xtest = [xtest]

			pp = clf.predict_proba(xtest)[0] #returns list of all possible label probabilities
			probaDict = dict(zip(clf.classes_,pp))				
			probaDict = {k:v for k,v in probaDict.items() if v>0}

			for item in probaDict:
				if item in overall_predictions_dict:
					overall_predictions_dict[item] += probaDict[item]
				else:
					overall_predictions_dict[item] = probaDict[item]
			if len(overall_predictions_dict)>30: break						

		print('Time for aggregating probabilities->{:f}'.format(round(time.time()-tNEW,10)),' seconds')

		tTemp = time.time()
		overall_predictions_dict = OrderedDict(sorted(overall_predictions_dict.items(),key=lambda x: x[1],reverse=True))
		print('Overall-Sorting->{:f}'.format(round(time.time()-tTemp,10)),' seconds')
		rank = 0
		netProb = sum(list(overall_predictions_dict.values()))

		tTemp = time.time()		
		for item in overall_predictions_dict:
			if rank<15:
				predLabels = item.split('##')
				prob = overall_predictions_dict[item]/netProb
				rank = rank + 1
				score = round(prob,5)
				overall_predictions.append(predLabels+[rank,score])
			else:
				break
		print('Ranking and Scoring Took ->{:f}'.format(round(time.time()-tTemp,10)),' seconds')
		predictionsReplied.append(overall_predictions)

	print('Prediction-Time->',time.time()-t1,' seconds')

	for ind,p in enumerate(predictionsReplied):
		print('Predictions For Input#',ind+1)
		for item in p: print(item)	

	responseData = {"predictions":predictionsReplied,"accuracy":acc,"precision":precision,"recall":recall,"fscore":fscore}
	return(json.dumps(responseData))	
   
loadModels()

if __name__=='__main__':
	app.run(port=4060,debug=True)
