import spacy
from nltk import Tree

import numpy as np

np.set_printoptions(threshold=np.inf)
en_nlp = spacy.load('en_core_web_lg')


### 原来的依赖树
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


###
class node:
    def __init__(self, id, text):
        self.id = id
        self.text = text
        self.sidelist = []

    def addtext(self, addtext):
        self.text = addtext + " " + self.text

    def seten(self, enname):
        self.text = enname

    def addside(self, sideid):
        self.sidelist.append(sideid)

    def __repr__(self):
        return "%s %s %s" % (self.id, self.text, self.sidelist)


class Side:
    def __init__(self, id, text):
        self.id = id
        self.text = text
        self.connect = []
        self.sidelist = []

    def addnode(self, nodeid):
        self.connect.append(nodeid)

    def addside(self, sideid):
        self.sidelist.append(sideid)

    def __repr__(self):
        return "%s %s%s" % (self.id, self.text, self.connect)


def isN(tag):
    flag = False
    if tag == "NN" or tag == "NNP" or tag == "NNPS" or tag == "NNS" or \
            tag == "FW" or tag == "CD" or tag == "VBG" or tag == "VBN" \
            or tag == "MD"  or tag == "JJ" or tag == "DT" \
            or tag == "RB"  or tag == "VBD" or tag == "VBP" \
            or tag == "JJS" or tag == "RBR":
        flag = True
    return flag


def isVI(tag):
    flag = False
    if tag == "VB" or tag == "VBD" or tag == "VBP" or \
            tag == "VBZ" or tag == "VBN" or tag == "CC" or tag == "IN" or tag == "VBG" \
            or tag == "RP" or tag == "RB" or tag == "JJ" or tag == "TO":
        flag = True
    return flag


def isnv(dep):
    flag = False
    if dep == "nsubj" or dep == "dobj":
        flag = True
    return flag


def creattree(sentence):
    en1name = sentence[sentence.find("<e1>") + 4:sentence.find("</e1>")]
    en2name = sentence[sentence.find("<e2>") + 4:sentence.find("</e2>")]
    sentence_no_blank = sentence.replace(" ", "").replace("<e1>", "").replace("<e2>", "")
    sentence_nlp = sentence.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")

    doc = en_nlp(sentence_nlp)
    wordlist = []
    dic_node = {}
    dic_side = {}
    # 依赖树
    # [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

    # for token in doc:
    #     print(token.i,'{0}({1}) <-- {2} -- {3}({4})'.format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))

    for token in doc:
        wordlist.append(token.text)
        id = token.i
        if isN(token.tag_) or token.text in en1name or token.text in en2name:
            dic_node[id] = node(id, token.text)
        if isVI(token.tag_):
            dic_side[id] = Side(id, token.text)

    e1_id, e2_id, id = 0, 0, 0
    while id < len(doc):
        token = doc[id]
        text = token.text
        head_token = token.head
        if sentence_no_blank.find("</e1>") == 0:
            e1_id = id - 1
            sentence_no_blank = sentence_no_blank.replace("</e1>", "")
        elif sentence_no_blank.find("</e2>") == 0:
            e2_id = id - 1
            sentence_no_blank = sentence_no_blank.replace("</e2>", "")
        sentence_no_blank = sentence_no_blank.replace(text, "", 1)
        # print(token.text, token.tag_, head_token.text, head_token.tag_)
        if isN(token.tag_):
            # if isN(head_token.tag_):
            # dic_node[head_token.i].addtext(token.text)
            if isVI(head_token.tag_):
                dic_node[id].addside(head_token.i)
                dic_side[head_token.i].addnode(id)
        if isVI(token.tag_):
            if isN(head_token.tag_):
                dic_node[head_token.i].addside(id)
                dic_side[id].addnode(head_token.i)
            if isVI(head_token.tag_):
                dic_side[head_token.i].addside(id)
                dic_side[id].addside(head_token.i)
        id += 1

    for id in range(len(doc)):
        token = doc[id]
        head_token = token.head
        if id == head_token.i:
            continue
        if isN(token.tag_):
            if isN(head_token.tag_):
                for sideson in dic_node[id].sidelist:
                    dic_node[head_token.i].addside(sideson)

    while isN(doc[e1_id].head.tag_) and not isnv(doc[e1_id].dep_) and doc[e1_id].head.i != e1_id:
        e1_id = doc[e1_id].head.i
    dic_node[e1_id].seten(en1name)

    while isN(doc[e2_id].head.tag_) and not isnv(doc[e2_id].dep_) and doc[e2_id].head.i != e2_id:
        e2_id = doc[e2_id].head.i
    dic_node[e2_id].seten(en2name)
    # print("e1_id :", e1_id, "e2_id :", e2_id)

    for key in dic_side.keys():
        if key in dic_node.keys() and key != e1_id and key != e2_id:
            del dic_node[key]
        if dic_side[key].sidelist != []:
            new_nodelist = []
            for i in dic_side[key].sidelist:
                new_nodelist += dic_side[i].connect
            dic_side[key].connect = list(set(new_nodelist))
        if len(dic_side[key].connect) >= 1:
            # print("side:", dic_side[key])
            continue

    for key in dic_node.keys():
        side_list = dic_node[key].sidelist
        new_side_list = side_list.copy()
        for side in side_list:
            new_side_list += dic_side[side].sidelist
        dic_node[key].sidelist = list(set(new_side_list))
        if dic_node[key].sidelist != []:
            # print("node:", dic_node[key])
            continue

    return dic_node, dic_side, e1_id, e2_id, wordlist


def creatnormal(dic_node, dic_side, e1_id, e2_id):
    new_dic_side = {}
    list_side = []
    i = 0
    pos_w2 = []
    for key in dic_side.keys():
        pos_w2.append(dic_side[key].id)
        new_dic_side[dic_side[key].id] = i
        list_side.append(dic_side[key].text)
        i += 1
    list_node = []
    A = []
    i = 0
    pos_w1 = []
    for key in dic_node.keys():
        if dic_node[key].sidelist == [] and key != e1_id and key != e2_id:
            continue
        pos_w1.append(dic_node[key].id)
        node_side = [0] * len(dic_side)
        for side in dic_node[key].sidelist:
            node_side[new_dic_side[side]] = 1
        list_node.append(dic_node[key].text)
        A.append(node_side)
        if key == e1_id:
            e1_id = i
        if key == e2_id:
            e2_id = i
        i += 1
    return list_node, list_side, A, e1_id, e2_id, pos_w1, pos_w2

def creatdata(dic_node,e1_id,e2_id):
    acc=np.zeros((128,128))
    e1=np.zeros((1,128))
    e1[0][e1_id]=1
    e2=np.zeros((1,128))
    e2[0][e2_id]=1

    for key in dic_node.keys():
        if dic_node[key].sidelist == []:
            continue
        nodeid=dic_node[key].id
        for sideid in dic_node[key].sidelist:
            if nodeid != sideid:
                acc[nodeid][sideid]=1
    
    acc_c=np.zeros((128,128))

    listfull=[e1_id]
    f1=e1_id
    while(listfull!=[])
    {

    }

    return acc,e1,e2


if __name__ == "__main__":
    sentence = "Stirring the hot <e1>popcorn</e1> around in a <e2>kettle</e2> drizzled with sugar, salt and oil evenly coats the popcorn."
    print(sentence)
    dic_node, dic_side, e1_id, e2_id, wordlist = creattree(sentence)
    
    # print(dic_side)
    # print(dic_node)
    acc ,e1 ,e2= creatdata(dic_node,e1_id,e2_id)
    print(wordlist,e1[:20],e2[:20])
    print(acc[:20][:20])
