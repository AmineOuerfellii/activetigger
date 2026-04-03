import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from scipy.spatial.distance import cdist

#--------------------------------------WAX-------------------------------------------------------------------#
#'Weistrassian Distance made explainable's paper implementation 2026
#
#
#
#
#
#------------------------------------------------------------------------------------------------------------#
  
def compute_sinkhorn(Xs: torch.Tensor, Yt: torch.Tensor, reg: float=0.01, num_iter: int=100, p: int=2) -> torch.Tensor:
  Xs_np = Xs.detach().numpy()
  Yt_np = Yt.detach().numpy()
  cost_matrix = cdist(Xs_np, Yt_np, metric='minkowski', p=p)
  K = np.exp(-cost_matrix / cost_matrix.max() / reg)
  n, m = K.shape
  mu = np.ones(n) / n
  nu = np.ones(m) / m
  u = np.ones(n)
  v = np.ones(m)
  for _ in range(num_iter):
      v = nu / (K.T @ u + 1e-10)
      u = mu / (K @ v + 1e-10)
  return torch.tensor(np.atleast_2d(u).T*(K*v.T),dtype=torch.float32)

class WaX():
  def __init__(self,X:torch.Tensor,Y:torch.Tensor,p:int,q:int,alpha:int,beta:int,n:int=100,reg :float=0.01):
    self.X = X.clone().requires_grad_(True)
    self.Y = Y.clone().requires_grad_(True)
    self.p = p
    self.q = q
    self.alpha = alpha
    self.beta = beta
    self.W=None
    #optimal transport
    self.gamma_star=compute_sinkhorn(Xs=X,Yt=Y,reg=reg,num_iter=n,p=self.q).detach()
    print("*gam",self.gamma_star)
    self.z_q=None
    self.z_beta=None
    self.z_kl=None
    #neutralization
    #constructing the neural network
    #Wasserstein distance
  def forward(self):
    self.z_q = torch.cdist(self.X,self.Y,p=self.q)
    # print("zq",self.z_q.isnan().any())
    self.z_beta=torch.cdist(self.X,self.Y,p=self.beta)
    # print("zz",self.z_beta.isnan().any())
    if self.beta == self.q:
      self.z_kl = self.z_q
    else:
      self.z_kl=self.z_beta * (self.z_q/self.z_beta).detach()
    # print("zzz",self.z_kl.isnan().any())
    # print(f"z_kl grad_fn:{self.z_kl.grad_fn}")
    W_p=torch.pow((self.gamma_star*(self.z_kl**self.p)).sum(),(1/self.p))
    # print("w_p",W_p)
    W_alpha=torch.pow((self.gamma_star*(self.z_kl**self.alpha)).sum(),(1/self.alpha))
    # print("w_alpha",W_alpha)
    self.W=W_alpha*(W_p/W_alpha).detach()
    # print("disatnc",self.W)
    return self.W
  def explain(self):
    self.X = self.X.detach().clone().requires_grad_(True)
    self.Y = self.Y.detach().clone().requires_grad_(True)
    self.forward()
    self.W.backward()
    Ri = (self.X*self.X.grad).sum(0) + (self.Y*self.Y.grad).sum(0)
    print(f"WaX done | W_p = {self.W.item():.6f} | sum(R_i) = {Ri.sum().item():.6f}")
    return Ri
  
class U_WaX:
  def __init__(self, Xs: torch.Tensor, Yt: torch.Tensor, r: int = 4, C: int = 3,lr: float = 0.01, n_iter: int = 200):
    self.d = Xs.shape[1]
    self.Xs = Xs.clone()
    self.Yt = Yt.clone()
    self.r = r
    self.C = C
    self.gamma_star = compute_sinkhorn(Xs, Yt, reg=0.01, num_iter=50).detach()
    U = torch.randn(self.d, self.d, device=Xs.device)
    self.U, _ = torch.linalg.qr(U)
    self.U.requires_grad_(True)
    optimizer = torch.optim.Adam([self.U], lr=lr)
    # ====================== OPTIMIZATION LOOP ======================
    for _ in range(n_iter):
        optimizer.zero_grad()
        diff = self.Xs.unsqueeze(1) - self.Yt.unsqueeze(0)        
        zc = torch.einsum('nmd,dc->nmc', diff, self.U)            
        Qc = torch.pow((self.gamma_star.unsqueeze(-1) * (zc.abs() ** self.r)).sum((0,1)), 1.0 / self.r)
        loss = -Qc[:self.C].sum()                                  
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.U.data = torch.linalg.qr(self.U.data)[0]
    self.U = self.U.detach()
  def explain(self):
    """Returns Rc and Rci (concept relevances + per-feature relevances)"""
    diff = self.Xs.unsqueeze(1) - self.Yt.unsqueeze(0)
    zc = torch.einsum('nmd,dc->nmc', diff, self.U[:, :self.C])
    Sc = (self.gamma_star.unsqueeze(-1) * zc**2).sum((0,1)).sqrt()
    W2 = Sc.norm(p=2)
    Rc = (Sc**2 / (Sc**2).sum()) * W2
    Rckl_num = self.gamma_star.unsqueeze(-1) * (zc**2)
    Rckl = Rckl_num / Rckl_num.sum((0,1), keepdim=True) * Rc.unsqueeze(0).unsqueeze(0)
    Rci = torch.zeros((self.C, self.d), device=self.Xs.device)
    for c in range(self.C):
        Uc = self.U[:, c:c+1]                                 
        hat_diff=(diff @ Uc) @ Uc.T                       
        num=diff * hat_diff
        den=num.sum(dim=-1, keepdim=True) + 1e-10
        weight=num/den
        Rci[c]=(weight * Rckl[..., c:c+1]).sum((0, 1))
    return Rc, Rci
  