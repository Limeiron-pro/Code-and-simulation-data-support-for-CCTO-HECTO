from pathlib import Path
import json,time,argparse
import numpy as np
from scipy.optimize import least_squares
from model import Model,unpack,BASE,PAIRS,write_csv
ROOT=Path(__file__).resolve().parent
CASES=['05_all_both']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',choices=CASES+['all'],default='05_all_both');ap.add_argument('--check-jacobian',action='store_true');args=ap.parse_args()
 cfg0=json.loads((ROOT/'fit_config.json').read_text());c=json.loads((ROOT/'data/R11_candidate.json').read_text());p=np.array(c['parameters_metal']);x0=np.r_[np.log(p[:,0]),p[:,1],c['charge_multiplier_of_R16']];allrows=[];allels=[];summary=[]
 for case in (CASES if args.case=='all' else [args.case]):
  cfg=json.loads(json.dumps(cfg0));withheld='HECTO' if case.endswith('CCTO_only') else 'CCTO' if case.endswith('HECTO_only') else None
  if withheld:cfg['group_weight']['R13_'+withheld]=0.
  if case.startswith('06'):cfg['group_weight'].update(R13_CCTO=4.,R13_HECTO=4.)
  m=Model(cfg);free=np.arange(15) if case.startswith('05') else np.array([4,5,11,12] if case.startswith('01') else [4,5,6,11,12,13]);scale=np.array([np.log(2) if i<7 else .03 for i in free]);prior_weight=.1
  lo=np.r_[np.log(np.full(7,10.)),[.12]*6+[.10],.45];hi=np.r_[np.log(np.full(7,1e7)),[.6]*6+[.4],.95];lo[6]=np.log(3806.540010934668);hi[6]=np.log(380654.0010934668);lo[1]=np.log(188.8093121415194);hi[1]=np.log(18880.93121415194);lo[8]=.2;hi[8]=.4
  def expand(z):
   x=x0.copy();x[free]=z;return x
  def fun(z):return np.r_[m.residual(expand(z)),prior_weight*(z-x0[free])/scale]
  def jac(z):return np.vstack([m.jac(expand(z))[:,free],np.diag(prior_weight/scale)])
  if args.check_jacobian:
   from scipy.optimize._numdiff import approx_derivative
   zz=x0[free].copy();zz[0]-=.1;num=approx_derivative(fun,zz);error=np.linalg.norm(jac(zz)-num)/np.linalg.norm(num);assert error<1e-5;print(case,'Jacobian relative difference',error,flush=True);continue
  out=ROOT/'results'/case;out.mkdir(parents=True,exist_ok=True);t=time.perf_counter();print('Starting',case,flush=True);old,oldels=m.metrics(x0,'R11_baseline')
  opt=least_squares(fun,np.clip(x0[free],lo[free]+1e-9,hi[free]-1e-9),jac=jac,bounds=(lo[free],hi[free]),x_scale='jac',max_nfev=1500,ftol=1e-9,xtol=1e-10,gtol=1e-7,verbose=1);x=expand(opt.x);p,q=unpack(x);rows,els=m.metrics(x,case)
  for rr in [old,rows,oldels,els]:
   for row in rr:row['R13_role']='heldout' if row['case'].startswith('R13') and row['material']==withheld else 'training' if row['case'].startswith('R13') else 'previous_training'
  write_csv(out/'case_metrics.csv',old+rows);write_csv(out/'element_metrics.csv',oldels+els);allrows+=old+rows;allels+=oldels+els
  info=dict(status='DIAGNOSTIC_CANDIDATE_NOT_APPROVED',case=case,optimizer_success=bool(opt.success),message=opt.message,nfev=opt.nfev,elapsed_seconds=time.perf_counter()-t,cost=float(opt.cost),parameters_metal=p.tolist(),pair_order=PAIRS,charge_multiplier_of_R16=q,charges_e=dict(zip(['Ca','Sr','Bi','Na','Cu','Ti','O'],(BASE*q).tolist())),free_parameter_indices=free.tolist(),R13_withheld_material=withheld,R13_roles_explained='The other R13 labels used in training; both used if withheld is null. Cross-validation uses labels already inspected, not a pristine prospective test.',regularization=dict(weight=prior_weight,logA_scale=float(np.log(2)),rho_scale_A=.03),near_bounds=[int(free[k]) for k in range(len(free)) if min(abs(opt.x[k]-lo[free[k]]),abs(opt.x[k]-hi[free[k]]))<1e-4])
  (out/'candidate.json').write_text(json.dumps(info,indent=2));write_csv(out/'parameters.csv',[dict(pair=s,A_eV=float(v[0]),rho_A=float(v[1]),C_eV_A6=float(v[2])) for s,v in zip(PAIRS,p)])
  lines=['# DIAGNOSTIC CANDIDATE; units metal; types Ca Sr Bi Na Cu Ti O.','pair_style buck/coul/long 10.0','pair_modify shift no tail no','pair_coeff * * 0.0 1.0 0.0','kspace_style ewald 1.0e-8']+[f'set type {i} charge {v:.15g}' for i,v in enumerate(BASE*q,1)]+[f'pair_coeff {i} 7 {v[0]:.15g} {v[1]:.15g} {v[2]:.15g}' for i,v in enumerate(p,1)];(out/'candidate_metal.inc').write_text('\n'.join(lines)+'\n')
  for group in sorted(set(m.groups)):
   rr=[r for r in rows if r['group']==group];bb=[r for r in old if r['group']==group];summary.append(dict(case=case,group=group,force_RMSE_R11=float(np.sqrt(np.mean([r['force_RMSE_eV_A']**2 for r in bb]))),force_RMSE_candidate=float(np.sqrt(np.mean([r['force_RMSE_eV_A']**2 for r in rr]))),energy_RMSE_R11_meV_atom=float(np.sqrt(np.mean([r['relative_energy_error_meV_atom']**2 for r in bb]))),energy_RMSE_candidate_meV_atom=float(np.sqrt(np.mean([r['relative_energy_error_meV_atom']**2 for r in rr]))),role=rr[0]['R13_role']))
  print(case,'cost',opt.cost,'seconds',info['elapsed_seconds'],flush=True)
 if summary:write_csv(ROOT/'results/comparison.csv',summary)
if __name__=='__main__':main()
