import csv
import operator
import copy
from random import shuffle
import numpy as np
from collections import Counter

berkeley = list(csv.reader(open('berp-POS-training.txt', "rt", encoding='utf-8'), delimiter = '\t'))
#shuffle(berkeley)

berkeley.insert(0, [100, "NL", "NL"])
i = 1
while i < len(berkeley):
    if len(berkeley[i]) == 0:
        berkeley[i] = [100, "NL", "NL"]
    i += 1    

#### Splitting the training and test data ####
split = int(0.8*len(berkeley))
#split = len(berkeley) - 20
trainData = copy.deepcopy(berkeley)

with open('assgn2-test-set.txt', 'r') as tabFile:
    tabFileArr = tabFile.read().splitlines()

testData = list(csv.reader(open('assgn2-test-set.txt', 'rt', encoding='utf-8'), delimiter = '\t'))
#testData = berkeley[split:]
testData.insert(0, [100, "NL"])

i = 1
while i < len(testData):
    if len(testData[i]) == 0:
        testData[i] = [100, "NL"]
    i += 1    
        
testDataNew = copy.deepcopy(testData)
# #### Removing the tags from test data ####
# i = 0
# while i < len(testData):
#     del testData[i][2]
#     i += 1      
 

    
#### Creating the baseline tagger ####
nlCount = 0;
baseline = {}    
for row in trainData:
    if row[1] not in baseline:
        baseline[row[1]] = {}
        baseline[row[1]][row[2]] = 1
    else:
        value = baseline[row[1]]
        if row[2] not in value:
            baseline[row[1]][row[2]] = 1
        else: 
            baseline[row[1]][row[2]] = baseline[row[1]][row[2]] + 1      

#### Figuring the tags for the test data ####    
for row in testData:
    if row[1] not in baseline:
        row.append("NN")
    else:
        fromBaseline = baseline[row[1]]
        maxKey = max(fromBaseline.items(), key=operator.itemgetter(1))[0]
        row.append(maxKey)

# #### Calculating the accuracy ####        
# i = 0
# correctCount = 0;
# while i < len(testData):
#     if testData[i][2] == testDataNew[i][2]:
#         correctCount += 1
#     i += 1
# print("Accuracy of baseline tagger: ", round(correctCount/len(testData)*100, 2))  

# i = 0
# while i < len(testData):
#     del testData[i][2]
#     i += 1  

##### Get Tags with counts ####

tags = {}
for row in trainData:
    if row[2] not in tags:
        tags[row[2]] = 1
    else:
        tags[row[2]] = tags[row[2]] + 1  

#### Build Tag transition probability matrix ####

tagTransitions = {}
i = 0
while i < len(trainData) - 1:
    if trainData[i][2] not in tagTransitions:
        tagTransitions[trainData[i][2]] = {}
        tagTransitions[trainData[i][2]][trainData[i+1][2]] = 1
    else:
        value = tagTransitions[trainData[i][2]]
        if trainData[i+1][2] not in value:
            tagTransitions[trainData[i][2]][trainData[i+1][2]] = 1
        else:
            tagTransitions[trainData[i][2]][trainData[i+1][2]] = tagTransitions[trainData[i][2]][trainData[i+1][2]] + 1
    i += 1   

#### Compute Bi-grams ####

bigrams = []
i = 0
while i < len(trainData) - 1:
    bigrams.append((trainData[i][2], trainData[i+1][2]))
    i += 1
uniqueBigrams = len(Counter(bigrams))

tagMat = np.zeros(shape=(len(tags), len(tags)))
tagKeys = list(tags.keys())

bigramKeys = Counter(bigrams).keys()

#### Compute frequency of words #####

words = []
for row in trainData:
    words.append(row[1])
    
wordCounts = Counter(words)

threshold = 1

#### Compute UNK frequency ####
 
UNK = 0    
for key, value in wordCounts.items():
    if value <= threshold:
        UNK += value
        
wordCounts.update({"UNK": UNK})  

#### Kneyser - Neys Smoothing ####
for row in range(0, len(tagKeys)):
    
    ### Compute lambda ####
    predCount = 0
    for value in bigramKeys:
        if value[0] == tagKeys[row]:
            predCount += 1       
    lamb = 0.75 / tags[tagKeys[row]] * predCount
    for col in range(0, len(tagKeys)):
        
        contProb = 0
        for value in bigramKeys:
            if(value[1] == tagKeys[col]):
                contProb += 1
        contProb = contProb/uniqueBigrams  
        
        #### Find Max ####
        combOccurrence = Counter(bigrams)[(tagKeys[row], tagKeys[col])]
        unigramOccurrence = tags[tagKeys[row]]
        
        tagMat[row][col] = max(combOccurrence - 0.75, 0)/unigramOccurrence + lamb * contProb

#### Build the emission probability matrix ####

emissions = {}
vocab = []
for tag in tags:
    for word in baseline:
        tagsForWord = baseline[word]
        if tag in tagsForWord:
            wordTagValue = tagsForWord[tag]
            tagValue = tags[tag]
            if tag not in emissions: 
                emissions[tag] = {}
            if wordCounts[word] <= threshold:
                if "UNK" not in emissions[tag]:
                    emissions[tag]["UNK"] = wordCounts[word]
                else:
                    emissions[tag]["UNK"] += wordCounts[word]
            else:        
                emissions[tag][word] = wordTagValue
                vocab.append(word)

vocab.append("UNK")

eMat = np.zeros(shape=(len(tags),len(vocab)))
for tag in tagKeys:
    value = emissions[tag]
    for word in vocab:
        if word in value:
            row = tagKeys.index(tag)
            col = vocab.index(word)
            eMat[row][col] = value[word]/tags[tag] 
            
            
# print("Vocab: ", vocab)
# print("Word counter: ", wordCounts)
# print("Tags: ", tags)
# print("Tag Keys: ", tagKeys)
# print("Emissions: ", emissions)
# print("Emission Matrix: ", eMat)
# print("Observation Matrix: ", tagMat)
# print("Bigrams: ", bigrams)
            
#### Viterbi Algorithm ####
 
numTest = len(testData)
numStates = tagMat.shape[0]
scale = np.zeros(numTest)

viterbi = np.zeros((numStates, numTest))
back = np.zeros((numStates, numTest))

reverse = np.zeros(numTest).astype(int)
npTest = np.array(testData)

for i in range(0, numStates):
    viterbi[i][0] = (1/numStates) * eMat[i][vocab.index(npTest[0][1])]
    if viterbi[i][0] != 0:
        print(viterbi[i][0])
    
scale[0] = 1/np.sum(viterbi[:,0])
viterbi[:,0] = scale[0] * viterbi[:,0]
back[0] = 0

avg = 0
sumScale = 0
for time in range (1, numTest):
    for state in range(0, numStates):
        transition = viterbi[:, time - 1] * tagMat[:, state]
        back[state, time], viterbi[state, time] = max(enumerate(transition), key = operator.itemgetter(1))
    
        if npTest[time][1] in vocab:
            viterbi[state][time] = viterbi[state][time] * eMat[state][vocab.index(npTest[time][1])]
        else:
            viterbi[state][time] = viterbi[state][time] * eMat[state][vocab.index("UNK")]
    if np.sum(viterbi[:, time]) != 0:
        scale[time] = 1.0/np.sum(viterbi[:, time])
    else:
        avg = sumScale/time
        scale[time] = avg
    sumScale += scale[time]      
    viterbi[:, time] = scale[time] * viterbi[:, time]   
    

verify = []    
reverse[numTest - 1] = np.argmax(viterbi[:, numTest - 1])  
verify.append(tagKeys[reverse[numTest - 1]])
for time in range(numTest - 1, 0, -1): 
    reverse[time - 1] = back[reverse[time], time]
    #print("Tag: ", tagKeys[reverse[time - 1]], " word: ", testData[time - 1][1])
    verify.append(tagKeys[reverse[time - 1]])
    
verify.reverse()

verify.pop(0)

for i in range(0, len(verify)):
    tabFileArr[i] += "\t"+verify[i]
    #print(tabFileArr)
    
#print(tabFileArr) 

with open("output.txt", "w") as f:
    for item in tabFileArr:
        if item == "\tNL":
            f.write("\n")
        else:
            f.write(item+"\n")
    

# accuracy = 0
# wrongCount = 0
# for i in range(0, numTest - 1):
#     if verify[i][1] == testDataNew[i][2]:
#         accuracy += 1
#     else:
#         #print("Word: ", testDataNew[i][1], "Correct Tag: ", testDataNew[i][2], "Associated Tag: ", verify[i][1])
#         wrongCount += 1
#         #print("Wrong count: ", wrongCount)
        
# print("Accuracy of Viterbi: ", round((accuracy/numTest)*100, 2))        
   



        
        

    




        
        



    
    
        

    
    