"""Fixed-charge local refinement. Uses exact periodic short-range differences
around actual R14 LAMMPS results, so no repeated Coulomb calculation is needed.
All 84 inspected labels are DEVELOPMENT data; no independent test claim.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from itertools import product
ROOT=Path(__file__).resolve().parent
ELS=['Ca','Sr','Bi','Na','Cu','Ti','O'];B=247

def prepare(records,params):
 # row: energy, 240 force components, 6 pressure components (GPa)
 pair=[]
 for el in ELS:
  radii=[];ec=[];er=[];fc=[];fr=[];fv=[];count=np.zeros(len(records))
  nedge=0
  for n,d in enumerate(records):
   H=np.array(d['H']);pos=np.array(d['positions']);pos=(pos@np.linalg.inv(H)%1)@H;sp=np.array(d['species']);V=abs(np.linalg.det(H))
   extent=np.ceil(10*np.linalg.norm(np.linalg.inv(H),axis=0)).astype(int)+1
   mask=((sp[:,None]==el)&(sp[None,:]=='O'))|((sp[:,None]=='O')&(sp[None,:]==el))
   ii,jj=np.where(mask)
   for shift in product(*[range(-int(v),int(v)+1) for v in extent]):
    dr=pos[ii]-pos[jj]-np.array(shift)@H;r=np.linalg.norm(dr,axis=1);take=(r>1e-8)&(r<10);r=r[take]
    if not len(r):continue
    i=ii[take];v=dr[take];col=np.arange(nedge,nedge+len(r));nedge+=len(r);radii.extend(r);ec.extend(col);er.extend([n*B]*len(r));count[n]+=.5*len(r)
    for k in range(3):fc.extend(col);fr.extend(n*B+1+3*i+k);fv.extend(v[:,k]/r)
    for k,(a,b) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):fc.extend(col);fr.extend([n*B+241+k]*len(r));fv.extend(.5*v[:,a]*v[:,b]/r/V*160.21766208)
  E=sparse.csr_matrix((np.full(len(ec),.5),(er,ec)),shape=(len(records)*B,nedge));FP=sparse.csr_matrix((fv,(fr,fc)),shape=E.shape)
  pair.append((np.array(radii),E,FP,count))
 return pair

def repulsion(x,pairs,p0,jac=False):
 size=len(pairs[0][3])*B;y=np.zeros(size);J=np.zeros((size,14)) if jac else None
 for k,(r,E,FP,count) in enumerate(pairs):
  A=p0[k,0]*np.exp(x[2*k]);rho=p0[k,1]+.02*x[2*k+1];v=A*np.exp(-r/rho);cut=A*np.exp(-10/rho)
  e=E@v;f=FP@(v/rho);y+=e+f;y[::B]-=count*cut
  if jac:
   J[:,2*k]=e+f;J[::B,2*k]-=count*cut
   J[:,2*k+1]=E@(v*r/rho**2*.02)+FP@(v*(r/rho**3-1/rho**2)*.02);J[::B,2*k+1]-=count*cut*10/rho**2*.02
 return (y,J) if jac else y

def vector(records,prefix):
 return np.concatenate([np.r_[d[prefix+'_E'],np.ravel(d[prefix+'_F']),d[prefix+'_P']] for d in records])

def objective_matrix(records,wcurve,wthermal):
 ri=[];ci=[];va=[];row=0
 def add(indices,weights):
  nonlocal row
  ri.extend([row]*len(indices));ci.extend(indices);va.extend(weights);row+=1
 base={d['material']:n for n,d in enumerate(records) if d['kind']=='base'}
 for n,d in enumerate(records):
  if d['kind'] not in ['base','thermal_holdout']:continue
  sp=np.array(d['species']);group='base' if d['kind']=='base' else 'thermal'
  # Equal material/element blocks; normalize sample count so plentiful species
  # do not erase minority A-site errors. Scales are declared fitting weights.
  for el in dict.fromkeys(sp):
   ids=np.where(sp==el)[0];weight=np.sqrt((1 if group=='base' else wthermal)/(2*len(set(sp))*(1 if group=='base' else 4)*len(ids)*3))/.3
   for i in ids:
    for k in range(3):add([n*B+1+3*i+k],[weight])
  w=np.sqrt(.3/(2*(1 if group=='base' else 4)*6))/.3
  for k in range(6):add([n*B+241+k],[w])
  if group=='thermal':
   b=base[d['material']];w=np.sqrt(.5/8)/(.02*80);add([n*B,b*B],[w,-w])
 groups={}
 for n,d in enumerate(records):
  if d['kind']=='displacement':groups.setdefault((d['material'],d['metadata']['pair']),[]).append(n)
 for ns in groups.values():
  assert len(ns)==2
  lo,hi=sorted(ns,key=lambda n:records[n]['metadata']['displacement_A']);h=records[hi]['metadata']['displacement_A']-records[lo]['metadata']['displacement_A'];meta=records[lo]['metadata'];atom=meta['atom_index_1']-1;ax=meta['axis']
  # One self curvature plus whole force-constant column; units eV/A^2.
  w=np.sqrt(wcurve/(len(groups)*2))/10
  add([hi*B+1+3*atom+ax,lo*B+1+3*atom+ax],[-w/h,w/h])
  w=np.sqrt(wcurve/(len(groups)*2*240))/3
  for k in range(240):add([hi*B+1+k,lo*B+1+k],[-w/h,w/h])
 shear=[n for n,d in enumerate(records) if d['kind']=='shear']
 for n in shear:
  b=base[records[n]['material']];w=np.sqrt(.5/len(shear))/.2
  # Stress increments; do not fit inconsistent CCTO shear-energy differences.
  add([n*B+244,b*B+244],[w,-w])
 return sparse.csr_matrix((va,(ri,ci)),shape=(row,len(records)*B))

def metrics(records,pred,df):
 rows=[]
 for n,d in enumerate(records):
  e=pred[n*B+1:n*B+241]-df[n*B+1:n*B+241]
  rows.append(dict(id=d['id'],material=d['material'],kind=d['kind'],T=d['metadata'].get('T'),force_RMSE_eV_A=float(np.sqrt(np.mean(e**2))),pressure_RMSE_GPa=float(np.sqrt(np.mean((pred[n*B+241:(n+1)*B]-df[n*B+241:(n+1)*B])**2)))))
 return rows

def candidate_inc(p):
 s='# R20 DEVELOPMENT ONLY. metal; types Ca Sr Bi Na Cu Ti O\npair_style buck/coul/long 10.0\npair_modify shift yes tail no table 0\npair_coeff * * 0.0 1.0 0.0\nkspace_style ewald 1.0e-8\n'
 for i,el in enumerate(ELS,1):s+=f"set type {i} charge {p['charges_e'][el]:.17g}\n"
 for i,(A,rho,C) in enumerate(p['parameters_metal'],1):s+=f'pair_coeff {i} 7 {A:.17g} {rho:.17g} {C:.17g}\n'
 return s

def main():
 records=json.loads((ROOT/'development_data.json').read_text());p=json.loads((ROOT/'baseline_parameters.json').read_text());p0=np.array(p['parameters_metal']);out=ROOT/'fit_results';out.mkdir(exist_ok=True)
 t=time.time();print('Preparing exact periodic pair features...',flush=True);pairs=prepare(records,p0);print('Prepared',round(time.time()-t,1),'s',flush=True)
 zero=np.zeros(14);r0=repulsion(zero,pairs,p0);md=vector(records,'R14');df=vector(records,'DFT')
 # Central finite differences independently verify analytic optimization Jacobian.
 y,J=repulsion(zero,pairs,p0,True)
 checks=[]
 for k in range(14):
  dx=zero.copy();dx[k]=1e-5;num=(repulsion(dx,pairs,p0)-repulsion(-dx,pairs,p0))/(2e-5);checks.append(float(np.max(abs(num-J[:,k]))/(1+np.max(abs(J[:,k])))))
 assert max(checks)<2e-6,checks
 summary=[]
 baseline=metrics(records,md,df);(out/'R14_metrics.json').write_text(json.dumps(baseline,indent=2))
 for name,wcurve,wthermal,reg in [('02_balanced',2.,1.,.25)]:
  L=objective_matrix(records,wcurve,wthermal);target=L@(df-md)
  def fun(x):return np.r_[L@(repulsion(x,pairs,p0)-r0)-target,reg*x]
  def jac(x):return np.vstack([L@repulsion(x,pairs,p0,True)[1],reg*np.eye(14)])
  fit=least_squares(fun,zero,jac=jac,bounds=(-np.ones(14),np.ones(14)),max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-8)
  pp=p0.copy();pp[:,0]*=np.exp(fit.x[::2]);pp[:,1]+=.02*fit.x[1::2];dest=out/name;dest.mkdir(exist_ok=True)
  candidate=dict(p);candidate.update(parameters_metal=pp.tolist(),status='DEVELOPMENT_CANDIDATE_NOT_APPROVED',charge_policy='fixed at R14; all C coefficients fixed',fit_weights=dict(curvature=wcurve,thermal=wthermal,regularization=reg),bounded_coordinates=fit.x.tolist(),optimizer_success=bool(fit.success),optimizer_message=fit.message,training_data_SHA256=hashlib.sha256((ROOT/'development_data.json').read_bytes()).hexdigest(),original_R14_training_data_included=False,limitations='Local refinement using 84 inspected R19 labels only. Original R14 training set is not bundled. Must recheck original training/offcenter cases before replacing R14. No pristine test or MD approval.')
  (dest/'parameters.json').write_text(json.dumps(candidate,indent=2));pred=md+repulsion(fit.x,pairs,p0)-r0;(dest/'metrics.json').write_text(json.dumps(metrics(records,pred,df),indent=2))
  (dest/'candidate.inc').write_text(candidate_inc(candidate))
  result=dict(candidate=name,success=bool(fit.success),nfev=fit.nfev,cost=float(fit.cost),initial_cost=float(.5*np.sum(fun(zero)**2)),near_bounds=[int(k) for k in np.where(abs(fit.x)>.98)[0]])
  summary.append(result);print(result,flush=True)
 (out/'fit_summary.json').write_text(json.dumps(dict(cases=summary,jacobian_relative_max_errors=checks,elapsed_s=time.time()-t,independent_test=False),indent=2))
 print('Balanced refinement reproduced; simulation inputs were not regenerated.',flush=True)
if __name__=='__main__':main()
