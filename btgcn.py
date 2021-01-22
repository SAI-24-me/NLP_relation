import math
import pickle
import warnings

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as py
from my_julei import *
# cuda版本不匹配warnnning

n_epochs = 60
learning_rate = 0.005
momentum = 0.5
random_seed = 1
batch_size = 400
torch.manual_seed(random_seed)


class GCN(nn.Module):
    def __init__(self, hid_size=256):
        super(GCN, self).__init__()

        self.hid_size = hid_size

        self.W = nn.Parameter(torch.FloatTensor(self.hid_size, self.hid_size).cuda())
        self.b = nn.Parameter(torch.FloatTensor(self.hid_size, ).cuda())

        self.init()

    def init(self):
        stdv = 1 / math.sqrt(self.hid_size // 2)

        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj, is_relu=True):
        out = torch.matmul(inp, self.W) + self.b
        out = torch.matmul(adj, out)

        if is_relu == True:
            out = nn.functional.relu(out)

        return out

gcn = GCN().cuda()

class Model_GraphRel(nn.Module):
    def __init__(self, num_rel=10,
                 hid_size=300, rnn_layer=2, gcn_layer=2, dp=0.5):
        super(Model_GraphRel, self).__init__()
        self.hid_size = hid_size
        self.gcn_layer = gcn_layer
        self.rnn_layer=rnn_layer
        self.num_rel=num_rel
        self.rnn = nn.GRU(300, self.hid_size,
                          num_layers=self.rnn_layer, batch_first=True, dropout=dp, bidirectional=True)
        self.gcn_w1 = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.gcn_layer)])
        self.gcn_w2 = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.gcn_layer)])

        self.w1_rel = nn.Linear(self.hid_size * 2, self.hid_size)
        self.w2_rel = nn.Linear(self.hid_size * 2, self.hid_size)
        self.pr_rel1 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.pr_rel2 = nn.Linear(self.hid_size, self.hid_size)
        self.pr_rel3  = nn.Linear(self.hid_size , self.num_rel)
        self.dp = nn.Dropout(dp)

    def forward(self, inp, pos_e1, pos_e2, dep_w12):
        out, _ = self.rnn(inp)
        # out = inp
        # out=inp
        # print(out.size())

        for i in range(self.gcn_layer):
            out =out + self.dp(self.gcn_w1[i](out, dep_w12.permute(0, 2, 1)))
            out =out + self.dp(self.gcn_w2[i](out, dep_w12))

        out_e1 = torch.matmul(pos_e1, out)
        out_e2 = torch.matmul(pos_e2, out)

        out_e1 = nn.functional.relu(self.w1_rel(out_e1))
        out_e1 = self.dp(out_e1)

        out_e2 = nn.functional.relu(self.w2_rel(out_e2))
        out_e2 = self.dp(out_e2)

        out_rel= torch.cat([out_e1, out_e2],dim=2)

        out_rel = nn.functional.relu(self.pr_rel1(out_rel))
        out_rel = nn.functional.relu(self.pr_rel2(out_rel))
        out_rel = self.pr_rel3(out_rel)
        out_rel = out_rel.view(-1,self.num_rel)

        return nn.functional.log_softmax(out_rel, dim=1)


net = Model_GraphRel().cuda()
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,momentum=momentum)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

def train(tvec,te1,te2,tacc,ttarget):
    net.train()
    optimizer.zero_grad()
    output = net(tvec,te1,te2,tacc)
    loss = nn.functional.nll_loss(output, ttarget)
    loss.backward()
    print(loss.item())
    optimizer.step()
    torch.save(optimizer.state_dict(), './optimizer.pth')


def test(tvec,te1,te2,tacc,ttarget):
    net.eval()
    with torch.no_grad():
        output = net(tvec,te1,te2,tacc)
        softmaxdata=output

        pred = output.data.max(1, keepdim=True)[1]
        return pred ,softmaxdata


train_data = pickle.load(open('./data/traindata.pkl', 'rb'))
test_data = pickle.load(open('./data/testdata.pkl', 'rb'))

for e in range(n_epochs):
    for i, (tvec,te1,te2,tacc,ttarget) in enumerate(train_data):
        train(tvec.cuda(),te1.cuda(),te2.cuda(),tacc.cuda(),ttarget.cuda())

    pre = []
    tru = []
    test_list = list()
    target_list=list()
    for i, (tvec,te1,te2,tacc,ttarget) in enumerate(test_data):
        prebath,softmaxdata= test(tvec.cuda(),te1.cuda(),te2.cuda(),tacc.cuda(),ttarget.cuda())
        test_list.append(softmaxdata)
        target_list.append(ttarget)
        prebath=prebath.cpu()
        ttarget=ttarget.cpu()
        pre.extend(prebath.numpy().tolist())
        tru.extend(ttarget.numpy().tolist())

    all_out = torch.cat(test_list, dim=0)
    all_out = all_out.cpu().numpy()
    target = torch.cat(target_list, dim=0)
    target = target.cpu().numpy()

    method = "KMeans"

    show_result(all_out, target, method)

    f1score = f1_score(tru, pre, average='macro' )
    acc= accuracy_score(tru, pre)
    print("f1:",f1score)
    print("acc:",acc)
