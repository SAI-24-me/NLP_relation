import linecache
import os
from pr1 import *
import pickle

def readallstr(i):
    txt = open(os.getcwd() + '/all_data.TXT', 'rb')
    data = txt.read().decode('utf-8')  # python3一定要加上这句不然会编码报错！
    txt.close()
    n = data.count('\n')
    # i = random.randint(1, (n + 1))
    i = i*4-3
    senten = linecache.getline(os.getcwd() + '/all_data.TXT', i)
    target = linecache.getline(os.getcwd() + '/all_data.TXT', i+1)
    target = target[:target.find(')')+1]
    num = senten[:senten.find('"')]

    senten = senten[senten.find('"') + 1:senten.rfind('"')]
    return int(num), senten ,target

def readtrainstr(i):
    txt = open(os.getcwd() + '/TRAIN_FILE.TXT', 'rb')
    data = txt.read().decode('utf-8')  # python3一定要加上这句不然会编码报错！
    txt.close()
    n = data.count('\n')
    # i = random.randint(1, (n + 1))
    i = i*4-3
    senten = linecache.getline(os.getcwd() + '/TRAIN_FILE.TXT', i)
    target = linecache.getline(os.getcwd() + '/TRAIN_FILE.TXT', i+1)
    target = target[:target.find(')')+1]
    num = senten[:senten.find('"')]

    senten = senten[senten.find('"') + 1:senten.rfind('"')]
    return int(num), senten ,target

def isconnecct(e1,e2,sidelist):
    slist=sidelist[e1].copy()
    nlist=sidelist[e1].copy()
    while True:
        for i in range (len(slist)):
            if slist[i]==0:
                continue
            for j in range(len(sidelist)):
                if sidelist[j][i]==1:
                    for k in range(len(slist)):
                        nlist[k]= nlist[k]|sidelist[j][k]
        if slist==nlist:
            break
        else:
            slist=nlist
    flag=0
    for i in range(len(slist)):
        flag+=slist[i]&sidelist[e2][i]
    if flag>0:
        return True
    else:
        return False



if __name__ == "__main__":
    sum_yes=0
    datalist=[]
    # for testi in range(1,10717):
    #     numid, sentence ,target= readallstr(testi)
    #     dic_node, dic_side, e1_id, e2_id, wordlist = creattree(sentence)
    #     list_node, list_side, A, e1_id, e2_id, pos_w1, pos_w2= creatnormal(dic_node, dic_side, e1_id, e2_id)
    #     if isconnecct(e1_id,e2_id,A):
    #         sum_yes+=1
    #         print(testi)
    #     else:
    #         print(sum_yes,testi)

    # sheng chen shu ju
    for testi in range(1,10717):
        if testi%100==0:
            print(testi)
        numid, sentence ,target= readallstr(testi)
        # print(numid, sentence)
        dic_node, dic_side, e1_id, e2_id, wordlist = creattree(sentence)
        acc ,e1 ,e2= creatdata(dic_node,e1_id,e2_id)
        if target.find("(e2,e1)")!=-1:
            target=target[:target.find("(")]
            datalist.append((wordlist,e2,e1,acc,target))
        else:
            target=target[:target.find("(")]
            datalist.append((wordlist,e1,e2,acc,target))
        # print(target)

    pickle.dump(datalist, open('./data/pre_data_test.pkl', 'wb'))