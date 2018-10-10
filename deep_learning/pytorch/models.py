import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

#语言模型LSTM
class LanguageModelLSTM(nn.Module):
    def __init__(self, vocab_size, class_size, em_sz, bs, nl, n_hidden):
        super().__init__()
        self.vocab_size,self.nl,self.class_size = vocab_size,nl,class_size
        self.bs,self.n_hidden = bs,n_hidden
        
        self.e = nn.Embedding(vocab_size, em_sz)
        self.rnn = nn.LSTM(em_sz, n_hidden, nl, dropout=0.5)
        self.l_out = nn.Linear(n_hidden, class_size)
        self.init_hidden()
        
    def forward(self, cs):
        bs = cs.size(1)
        if bs!=self.bs:
            self.bs = bs
            self.init_hidden()
        
        outp,h = self.rnn(self.e(cs), self.h)
        self.h = self.repackage_var(h)  #解除h中绑定的运算图
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, bs, self.class_size)
    
    def init_hidden(self):
        bs = self.bs
        
        if next(self.parameters()).is_cuda:
            self.h = (Variable(torch.zeros(self.nl, bs, self.n_hidden)).cuda(),
                  Variable(torch.zeros(self.nl, bs, self.n_hidden)).cuda())
        else:
            self.h = (Variable(torch.zeros(self.nl, bs, self.n_hidden)),
                  Variable(torch.zeros(self.nl, bs, self.n_hidden)))     
        
    def repackage_var(self,h):
        """Wraps h in new Variables, to detach them from their history."""
        if next(self.parameters()).is_cuda:
            return tuple(self.repackage_var(v) for v in h) if type(h) == tuple else Variable(h.data).cuda()
        else:
            return tuple(self.repackage_var(v) for v in h) if type(h) == tuple else Variable(h.data)

#分类用的LSTM
class ClassifyLSTM(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.vocab_size,self.nl,self.class_size = m.vocab_size,m.nl,m.class_size
        self.n_hidden = m.n_hidden
        self.bs = m.bs
        self.bs_old = m.bs
        
        self.e = nn.Embedding(m.e.weight.data.size(0), m.e.weight.data.size(1))
        self.rnn = nn.LSTM(m.e.weight.data.size(1), self.n_hidden, self.nl, dropout=0.5)
        self.l_out = nn.Linear(self.n_hidden, self.class_size)
        
        self.e.weight.data.copy_(m.e.weight.data)
        for w1,w2 in zip(self.rnn.parameters(),m.rnn.parameters()):
            w1.data.copy_(w2.data)
            
        self.l_out.weight.data.copy_(m.l_out.weight.data)
              
        
        self.init_hidden()
        
    def restore(self):
        self.bs = self.bs_old
        
    def forward(self, cs, end_ids, sqz_ids):
        self.h = self.change_hidden(end_ids,sqz_ids)
        if self.bs!=cs.size(1):
            print(f'batch size is different self.bs:{self.bs} x.bs:{cs.size(1)}!')
            self.bs = cs.size(1)
            self.init_hidden()
        
        outp,h = self.rnn(self.e(cs), self.h)
        #self.h = self.repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.bs, self.class_size)
    
    def init_hidden(self):
        
        if next(self.parameters()).is_cuda:
            self.h = (Variable(torch.zeros(self.nl, self.bs, self.n_hidden)).cuda(),
                  Variable(torch.zeros(self.nl, self.bs, self.n_hidden)).cuda())
        else:
            self.h = (Variable(torch.zeros(self.nl, self.bs, self.n_hidden)),
                  Variable(torch.zeros(self.nl, self.bs, self.n_hidden)))     

    #自适应动态batch     
    def change_hidden(self,end_ids,sqz_ids):
        
        end_ids = end_ids.cpu()
        sqz_ids = sqz_ids.cpu()
        
        h0 = torch.zeros_like(self.h[0].data.cpu())
        h1 = torch.zeros_like(self.h[1].data.cpu())
        
        h0.copy_(self.h[0].data.cpu())
        h1.copy_(self.h[1].data.cpu())
        if len(end_ids)>0:
            h0[:,end_ids,:] = 0
            h1[:,end_ids,:] = 0
            
                
        if len(sqz_ids)>0:
            left_ids = [i for i in range(h0.size(1)) if i not in set(sqz_ids)]
            h0 = h0[:,left_ids,:].contiguous()
            h1 = h1[:,left_ids,:].contiguous()
            
            self.bs -= len(sqz_ids)
            
        if next(self.parameters()).is_cuda:
            return (Variable(h0).cuda(),Variable(h1).cuda())
        else:
            return (Variable(h0),Variable(h1))    
            
    #解除变量上绑定的计算图          
    def repackage_var(self,h):
        """Wraps h in new Variables, to detach them from their history."""
        if next(self.parameters()).is_cuda:
            return tuple(self.repackage_var(v) for v in h) if type(h) == tuple else Variable(h.data).cuda()
        else:
            return tuple(self.repackage_var(v) for v in h) if type(h) == tuple else Variable(h.data)