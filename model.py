import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math

class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):

        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.W = nn.Linear(768, emb_dimension)
        self.init_emb()

    def qk_net(self, q_, k_):
        att = F.softmax(torch.sum(torch.mul(q_, k_).div(math.sqrt(self.emb_dimension)).squeeze(), dim=1).unsqueeze(0), dim=1).squeeze(0)  # 10
        return att

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(0, 0)
        torch.nn.init.xavier_uniform_(self.W.weight, gain=1)

    def forward(self, pos_u, pos_v, neg_v, u_bert):

        # emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v) #50*100
        emb_ubert = self.W(torch.cat(u_bert).cuda())#50*768
        # emb_ubert = torch.cat(u_bert).cuda()
        # score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.mul(emb_ubert, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        att = self.qk_net(emb_ubert, emb_v) #1*50
        score = torch.mul(score,att)#
        score = F.logsigmoid(score)

        neg_emb_v = self.v_embeddings(neg_v)
        # neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.bmm(neg_emb_v, emb_ubert.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score)+torch.sum(neg_score))

    def save_embedding(self, id2word, file_name):

        # embedding = self.u_embeddings.weight.cpu().data.numpy()
        #
        # fout = open(file_name, 'w')
        # # fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        # for wid, w in id2word.items():
        #     e = embedding[wid]
        #     e = ' '.join(map(lambda x: str(x), e))
        #     fout.write('%s %s\n' % (w, e))
        #

        embedding2 = self.v_embeddings.weight.cpu().data.numpy()
        fout = open(file_name + '_v', 'w')
        # fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding2[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

