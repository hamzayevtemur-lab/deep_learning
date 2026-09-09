import torch 
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm1d(nn.Module):
    def __init__(self, dim, momentum=0.1):
        super().__init__()
        self.momentum=momentum
        self.eps=1e-6
        self.scale=nn.Parameter(torch.ones(dim))
        self.shift=nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        
    def forward(self, x):
        if self.training:
            xmean=x.mean(dim=0, keepdim=True)
            xvar=x.var(dim=0, keepdim=True)
            
            with torch.no_grad():
                self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*xmean.squeeze(0)
                self.running_var=(1-self.momentum)*self.running_var+self.momentum*xvar.squeeze(0)
                
        else:
            xmean=self.running_mean
            xvar=self.running_var
            
        x=(x-xmean)/(xvar+self.eps)**0.5
        
        return self.scale*x +self.shift
    
    
class NameGenerator(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_hidden, n_layers, dropout=0.2):
        super().__init__()
        self.block_size=block_size
        
        self.E=nn.Embedding(vocab_size, n_embd)
        self.input_proj=nn.Linear(block_size*n_embd, n_hidden, bias=False)
        self.input_bn=BatchNorm1d(n_hidden)
        self.drop=nn.Dropout(dropout)
        
        self.layers=nn.ModuleList([
            nn.Linear(n_hidden, n_hidden, bias=False) for _ in range(n_layers)
        ])
        
        self.bns=nn.ModuleList([
            BatchNorm1d(n_hidden) for _ in range(n_layers)
        ])
        
        self.out=nn.Linear(n_hidden, vocab_size)
        
        nn.init.xavier_uniform_(self.input_proj.weight)
        
        for l in self.layers:
            nn.init.xavier_uniform_(l.weight)
            
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        
    
    def forward(self, x):
        x=self.E(x)
        x=x.view(x.size(0), -1)
        x=torch.tanh(self.input_bn(self.input_proj(x)))
        x=self.drop(x)
        
        for layer , bn in zip(self.layers, self.bns):
            x=torch.tanh(bn(layer(x)))
            x=self.drop(x)
        
        return self.out(x)
    
    
    @torch.no_grad()
    def generate(self, n, temperature, itos):
        self.eval()
        results=[]
        
        for _ in range(n):
            context=[0]*self.block_size
            chars=[]
            
            for _ in range(20):
                x=torch.tensor([context]).to(next(self.parameters()).device)
                probs=F.softmax(self(x)/temperature, dim=-1)
                ix=torch.multinomial(probs, 1).item()
                
                if ix==0:
                    break
                
                chars.append(itos[ix])
                context=context[1:]+[ix]
                
            results.append("".join(chars))
            
        return results
            
            

    