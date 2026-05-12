from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.rl.replay_buffer import ReplayBuffer

class QNet(nn.Module):
    def __init__(self,obs_dim:int,n_actions:int,dueling:bool=False):
        super().__init__(); self.dueling=dueling
        self.back=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        if dueling:
            self.v=nn.Linear(128,1); self.a=nn.Linear(128,n_actions)
        else: self.q=nn.Linear(128,n_actions)
    def forward(self,x,mask=None):
        h=self.back(x)
        if not self.dueling: return self.q(h)
        v=self.v(h); a=self.a(h)
        if mask is None: return v+a-a.mean(dim=1,keepdim=True)
        m=mask.float(); denom=torch.clamp(m.sum(dim=1,keepdim=True),min=1.0)
        a_mean=(a*m).sum(dim=1,keepdim=True)/denom
        return v+a-a_mean

class DQNAgent:
    def __init__(self,obs_dim,n_actions,cfg,method='proposed'):
        self.n_actions=n_actions; self.method=method
        dueling=method=='proposed'; self.masked=method in {'am_ddqn_dr','proposed'}; self.double=method in {'ddqn_dr','am_ddqn_dr','proposed'}
        self.online=QNet(obs_dim,n_actions,dueling); self.target=QNet(obs_dim,n_actions,dueling); self.target.load_state_dict(self.online.state_dict())
        self.opt=optim.Adam(self.online.parameters(),lr=cfg['lr']); self.gamma=cfg['gamma']; self.eps=cfg['epsilon_start']; self.eps_end=cfg['epsilon_end']; self.eps_decay=cfg['epsilon_decay']
        self.rb=ReplayBuffer(cfg['replay_capacity']); self.bs=cfg['batch_size']
    def select_action(self,obs,action_mask,info=None):
        if np.random.rand()<self.eps: return int(np.random.randint(0,self.n_actions))
        with torch.no_grad():
            q=self.online(torch.tensor(obs,dtype=torch.float32).unsqueeze(0), torch.tensor(action_mask).unsqueeze(0) if self.masked else None)[0].numpy()
        if self.masked: q=np.where(np.asarray(action_mask)>0,q,-1e9)
        return int(np.argmax(q))
