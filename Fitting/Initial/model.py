"""R6 fixed-charge Buckingham refit. Python 3.9+, NumPy, SciPy. Metal units."""
from pathlib import Path
import json,csv,time,argparse,hashlib
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parent
PAIRS=['Ca-O','Sr-O','Bi-O','Na-O','Cu-O','Ti-O','O-O']
P0=np.array([[82076.80502168658,.1800000000000428,0],[94404.65607075968,.1859785635442092,0],[1565.93635421665,.3091123914104315,0],[9873.486175921553,.18,0],[19746.97235184311,.18,0],[19680.34887066812,.1990374421118951,0],[57259.38416818096,.1200000000000001,20.76395908872267]])
Q0=.7083869037262285
BASE=np.array([2,2,3,1,2,4,-2])*.828692482570312
PRESS=[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
def unpack(x):
 p=P0.copy();p[:,0]=np.exp(x[:7]);p[:,1]=x[7:14];return p,float(x[14])
class Model:
 def __init__(self,cfg):
  self.cfg=cfg;self.refs=json.loads((ROOT/'data/training_references.json').read_text());z=np.load(ROOT/'data/fit_geometry.npz');self.C=np.array([z[f'coul_{i}'] for i in range(len(self.refs))]);self.g=[[(z[f'indices_{i}_{j}'],z[f'vectors_{i}_{j}'],z[f'distances_{i}_{j}']) for j in range(7)] for i in range(len(self.refs))];self.V=[r['volume_A3'] for r in self.refs]
  self.target=np.array([np.r_[r['energy_eV'],np.array(r['forces_eV_A']).ravel(),r['pressure_GPa']] for r in self.refs]);self.anchors=np.array([next(i for i,s in enumerate(self.refs) if s['material']==r['material'] and s['tag']=='00_equilibrium') for r in self.refs])
  groups=[('R13' if r['tag'].startswith('R13') else 'R10' if r['tag'].startswith('R10') else 'R5' if r['tag'].startswith('R5') else 'R4' if r['tag'].startswith('R4') else 'original')+'_'+r['material'] for r in self.refs]
  self.groups=groups;self.caseweight=np.array([cfg['group_weight'][s]/np.sqrt(groups.count(s)) for s in groups]);self.weights=np.r_[1/(cfg['energy_scale_eV_atom']*80),np.full(240,1/cfg['force_scale_eV_A']/np.sqrt(240)),np.full(6,1/cfg['pressure_scale_GPa']/np.sqrt(6))]
  self.lastx=None
 def evaluate(self,x):
  if self.lastx is not None and np.array_equal(x,self.lastx):return self.pred,self.der
  p,q=unpack(x);pred=q*q*self.C;J=np.zeros((len(self.refs),247,15));J[:,:,14]=2*q*self.C
  for n,(geom,V) in enumerate(zip(self.g,self.V)):
   def add(e,f,col=None):
    out=pred[n] if col is None else J[n,:,col];out[0]+=e.sum()/2
    for k in range(3):out[1+k:241:3]+=np.bincount(idx,weights=f[:,k],minlength=80)
    out[241:]+=np.array([(f[:,i]*dr[:,j]).sum() for i,j in PRESS])*.5/V*160.21766208
   for j,((idx,dr,r),(A,rho,C)) in enumerate(zip(geom,p)):
    ex=A*np.exp(-r/rho);unit=dr/r[:,None]
    add(ex-C/r**6,(ex/rho-6*C/r**7)[:,None]*unit)
    add(ex,(ex/rho)[:,None]*unit,j)
    add(ex*r/rho**2,(ex*(r/rho**3-1/rho**2))[:,None]*unit,j+7)
  self.lastx=x.copy();self.pred=pred;self.der=J;return pred,J
 def residual_jac(self,x):
  pred,J=self.evaluate(x);err=pred-self.target;JJ=J.copy();err[:,0]-=(pred[self.anchors,0]-self.target[self.anchors,0]);JJ[:,0]-=J[self.anchors,0]
  weight=self.caseweight[:,None]*self.weights[None,:];p,q=unpack(x);r=np.array([1.5]*6+[1.0]);ex=p[:,0]*np.exp(-r/p[:,1]);coul=14.3996454784255*BASE*BASE[-1]/r**2
  force=ex/p[:,1]-6*p[:,2]/r**7+q*q*coul;active=force<0;guards=np.minimum(force,0)/self.cfg['guard_scale_eV_A'];G=np.zeros((7,15))
  for j in range(7):
   if active[j]:G[j,j]=ex[j]/p[j,1];G[j,j+7]=ex[j]*(r[j]/p[j,1]**3-1/p[j,1]**2);G[j,14]=2*q*coul[j]
  G/=self.cfg['guard_scale_eV_A']
  return np.r_[(err*weight).ravel(),guards],np.vstack([(JJ*weight[:,:,None]).reshape(-1,15),G])
 def residual(self,x):return self.residual_jac(x)[0]
 def jac(self,x):return self.residual_jac(x)[1]
 def metrics(self,x,stage):
  pred,_=self.evaluate(x);rows=[];elements=[]
  for i,r in enumerate(self.refs):
   f=(pred[i,1:241]-self.target[i,1:241]).reshape(80,3);a=self.anchors[i]
   rows.append(dict(stage=stage,material=r['material'],case=r['tag'],group=self.groups[i],force_RMSE_eV_A=float(np.sqrt(np.mean(f*f))),pressure_RMSE_GPa=float(np.sqrt(np.mean((pred[i,241:]-self.target[i,241:])**2))),DFT_relative_energy_meV_atom=float((self.target[i,0]-self.target[a,0])*12.5),MD_relative_energy_meV_atom=float((pred[i,0]-pred[a,0])*12.5),relative_energy_error_meV_atom=float(((pred[i,0]-pred[a,0])-(self.target[i,0]-self.target[a,0]))*12.5)))
   for el in sorted(set(r['species'])):
    sel=np.array(r['species'])==el;elements.append(dict(stage=stage,material=r['material'],case=r['tag'],element=el,force_RMSE_eV_A=float(np.sqrt(np.mean(f[sel]**2)))))
  return rows,elements

def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)