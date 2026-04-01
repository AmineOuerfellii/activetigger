from ..XAi.wax import WaX, U_WaX
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.compute_sbert import ComputeSbert
import torch
import pandas as pd
from pathlib import Path

class ComputeWaXUWaX(BaseTask):
    kind = "compute_wax_uwax"
    def __init__(self,source_texts:list[str],target_texts:list[str],path_process:Path,model:str="all-MiniLM-L6-v2",p:int=2,q:int=2,alpha:int=2,beta:int=2,n:int=100,reg:float=0.01,r:int=4,C:int=3,lr:float=0.01,n_iter:int=200):
        super().__init__()
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.path_process = path_process
        self.model = model
        self.p = p
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.reg = reg
        self.r = r
        self.C = C
        self.lr = lr
        self.n_iter = n_iter
    def __call__(self):
        # Embed source and target texts using ComputeSbert
        #print(self.source_texts.shape, self.target_texts.shape)
        source_series = pd.Series(self.source_texts)
        target_series = pd.Series(self.target_texts)
        csbert_source = ComputeSbert(texts=source_series,path_process=self.path_process,model=self.model)
        csbert_target = ComputeSbert(texts=target_series,path_process=self.path_process,model=self.model)
        X = torch.tensor(csbert_source().values, dtype=torch.float32)
        Y = torch.tensor(csbert_target().values, dtype=torch.float32)
        wax = WaX(X, Y, self.p, self.q, self.alpha, self.beta, self.n, self.reg)
        wax.forward()
        gamma = wax.gamma_star          # shape (N, M)
        z_kl = wax.z_kl                 # shape (N, M)
        numerator = gamma * (z_kl ** self.alpha)
        denominator = numerator.sum()
        R_kl = numerator / denominator * wax.W   # Eq. (3a) from paper
        instance_relevance = R_kl.sum(axis=0).tolist()
        Ri = wax.explain()
        #leave for confirmation of pull request
        #uwax = U_WaX(X, Y, self.r, self.C, self.lr, self.n_iter)
        #Rc, Rci = uwax.explain()
        return {"wax_distance": wax.W.item(),"feature_relevance": Ri.tolist(),"instance_relevance":instance_relevance}