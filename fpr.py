import numpy as np
import pickle
import torch
pre2_data = pickle.load(open('./data/pre2_data.pkl', 'rb'))

batch_size=400

traindata=[]
testdata=[]

tmpvec=[]
tmpe1=[]
tmpe2=[]
tmpacc=[]
tmptargetn=[]

for i, (vec,e1, e2, acc, targetn) in enumerate(pre2_data):
    tmpvec.append(vec)
    tmpe1.append(e1)
    tmpe2.append(e2)
    tmpacc.append(acc)
    tmptargetn.append(targetn-1)
    
    if (i+1)%batch_size==0:
        tvec=torch.Tensor(tmpvec)
        te1=torch.Tensor(tmpe1)
        te2=torch.Tensor(tmpe2)
        tacc=torch.Tensor(tmpacc)
        ttarget=torch.LongTensor(tmptargetn)
        if i <=8000:
            traindata.append((tvec,te1,te2,tacc,ttarget))
        else:
            testdata.append((tvec,te1,te2,tacc,ttarget))

        tmpvec=[]
        tmpe1=[]
        tmpe2=[]
        tmpacc=[]
        tmptargetn=[]

if len(tmpvec)>0:
    tvec=torch.Tensor(tmpvec)
    te1=torch.Tensor(tmpe1)
    te2=torch.Tensor(tmpe2)
    tacc=torch.Tensor(tmpacc)
    ttarget=torch.LongTensor(tmptargetn)
    testdata.append((tvec,te1,te2,tacc,ttarget))

pickle.dump(traindata, open('./data/traindata.pkl', 'wb'))
pickle.dump(testdata, open('./data/testdata.pkl', 'wb'))