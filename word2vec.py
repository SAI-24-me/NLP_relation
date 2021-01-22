import numpy as np
import pickle
glove_path = r'./glove.840B.300d.txt'
senten =['the', 'weather', 'be', 'still', 'the', 'main', 'talking', 'point', 'on', 'the', 'front', 'page', '.']
num2vec={}
i = 1
with open(glove_path, 'rb') as f:
    for line in f:
        if i%10000==0:
            print(i)
            # break
        i+=1
        line = line.decode().split()
        while "." in line :
            line[line.index(".")]="0.0"
        word = line[0]
        try:
            vect = np.array(line[1:]).astype(np.float)
            assert vect.size == 300
        except ValueError :
            pass
        except AssertionError :
            pass
        else:
            num2vec[word]=vect

num2vec["$"]=np.zeros((300,))
pre_data = pickle.load(open('./data/pre_data_test.pkl', 'rb'))
targetdic={}
noneword={}
targetnum=1
pre2_data=[]
print("down")
for i, (wlist, e1, e2, acc, target) in enumerate(pre_data):

    if target not in targetdic.keys():
        targetdic[target]=targetnum
        targetnum+=1
    
    targetn=targetdic[target]

    vec=[]
    for w in wlist:
        if  w in num2vec.keys():
            vec.append(num2vec[w])
        else:
            if w not in noneword.keys():
                noneword[w]=np.random.rand(300)
            vec.append(noneword[w])
    padnum=len(vec)
    while padnum<128:
        padnum+=1
        vec.append(num2vec["$"])
    pre2_data.append((vec,e1, e2, acc, targetn))

    if i%100==0:
        print(i)

print("down")
print("noword:",len(noneword.keys()))
pickle.dump(pre2_data, open('./data/pre2_data.pkl', 'wb'))
    
    