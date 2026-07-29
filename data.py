import numpy as np


# encoder is used to convert characters into numbers, which can be understood by the model
def encoder(inputString,stoi):
    finalEncodeList =[]
    for i in inputString:
        finalEncodeList.append(stoi[i])
    return finalEncodeList


# decoder is used to convert the numbers into characters, which can be understood by the user
def decoder(inputList,itos):
    finalDecodedString =''
    for i in inputList:
        finalDecodedString=finalDecodedString+itos[i]
    return finalDecodedString


''' 
. this function creates contextStack and targetStack, according to the blockSize and batchSize
. batchSize is used to get more contextStack for model to train in one go instead of training 
  on one contextStack which is slower, so we use batch to train more context windows in one go
. inputTextLen is the length of the input text
. blockSize is the size of the one context window, bigger blockSize means more history to 
  look at but it takes more memory and computational power
. data is the numpy array which stores the numerical values of each character from the input text
. offset is created to get the next character 
'''
def getBatch(batchSize, inputDataLen, blockSize, data, offset):
    contextList =[]
    targetList =[]
    for _ in range (batchSize):
        startingPoint = np.random.randint(0,inputDataLen-blockSize-1)
        x= data[startingPoint:startingPoint+blockSize]
        y= data[startingPoint+offset: startingPoint+blockSize+offset]
        contextList.append(x)
        targetList.append(y)    
    contextStack = np.stack(contextList)
    targetStack = np.stack(targetList)
    return contextStack, targetStack

# to open a file
with open('input.txt','r', encoding='utf-8') as inputFile:
    text = inputFile.read()


# set() removes duplicates, sorted() orders them into a list
chars = sorted(set(text))

# how many unique characters there are
vocabSize = len(chars)  

# print the vocabulary as one clean line, and its size
# join stitches the list of characters into a single string
# print("Vocabulary: ")
# print(''.join(chars))   

# to get the the vocabulary size
# print("Vocabulary size: ")
# print(vocabSize)

# through stoi you get numbers using char (char = key, number = value)
stoi = {}

# through itos you get char using numbers (char = value, number = key)
itos = {}


# this stores the key value pair for stoi and itos
for count,i in enumerate(chars):
    stoi[i] = count
    itos[count] = i


# converting entire input text into is numerical equivalent 
data = np.array(encoder(text,stoi))

#to get the lenght of the input text
inputDataLen =len(text)


# the total number of characters in one context window
blockSize = 8

# to get the target value on the basis of context window
offset= 1

# total number of batches to train in one go
batchSize = 4


# to get the context and target batch
x,y = getBatch(batchSize, inputDataLen, blockSize, data, offset)

print("Context Stack: ",x)
print("Target Stack: ",y)
print("Context Shape: ",x.shape)
print("Target Shape: ", y.shape)

# for t in range(blockSize):
#     context =decoder(x[:t+1], itos)
#     target = itos[y[t]]
#     print("Context is: ",context," target is: ", target)