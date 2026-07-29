import numpy as np
def encoder(inputString,stoi):
    finalEncodeList =[]
    for i in inputString:
        finalEncodeList.append(stoi[i])
    return finalEncodeList

def decoder(inputList,itos):
    finalDecodedString =''
    for i in inputList:
        finalDecodedString=finalDecodedString+itos[i]
    return finalDecodedString

# to open a file
with open('input.txt','r', encoding='utf-8') as inputFile:
    text = inputFile.read()

chars = sorted(set(text))    # set() removes duplicates, sorted() orders them into a list
vocab_size = len(chars)      # how many unique characters there are

# print the vocabulary as one clean line, and its size
# print("Vocabulary: ")
print(''.join(chars))        # join stitches the list of characters into a single string
# print("Vocabulary size: ")
print(vocab_size)

stoi = {} 
# through stoi you get numbers using char (char = key, number = value)
itos ={}
# through itos you get char using numbers (char = value, number = key)
count=0
for count,i in enumerate(chars):
    stoi[i] = count
    itos[count] = i


data = np.array(encoder(text,stoi))
# print(data.shape)
# print(data[:100])
# print(decoder([46,47],itos))

block_size = 8
offset= 1
x= data[:block_size]
y= data[1:block_size+offset]
print(x)
print(y)

for t in range(block_size):
    context =decoder(x[:t+1], itos)
    target = itos[y[t]]
    print("Context is: ",context," target is: ", target)