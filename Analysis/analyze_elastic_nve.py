"""Standard-library only: collect all completed and failed jobs, no reruns."""
from pathlib import Path
import json,csv,re,math,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]/'Validation'
VOIGT=['xx','yy','zz','yz','xz','xy']
def csvwrite(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def eig_sym(a):
 a=[r[:] for r in a];n=len(a)
 for _ in range(1000):
  p,q=max(((p,q) for p in range(n) for q in range(p+1,n)),key=lambda pq:abs(a[pq[0]][pq[1]]))
  if abs(a[p][q])<1e-10:break
  phi=.5*math.atan2(2*a[p][q],a[q][q]-a[p][p]);c=math.cos(phi);s=math.sin(phi);ap=a[p][p];aq=a[q][q];v=a[p][q]
  for k in range(n):
   if k not in [p,q]:
    x,y=a[k][p],a[k][q];a[k][p]=a[p][k]=c*x-s*y;a[k][q]=a[q][k]=s*x+c*y
  a[p][p]=c*c*ap-2*s*c*v+s*s*aq;a[q][q]=s*s*ap+2*s*c*v+c*c*aq;a[p][q]=a[q][p]=0.
 return sorted(a[k][k] for k in range(n))
def main():
 rows=[];tensor=[];elastic=[]
 for t in json.loads((ROOT/'tasks.json').read_text()):
  d=ROOT/t['folder'];row={k:t[k] for k in ['folder','candidate','material','kind','natoms']};row['status']='PENDING'
  try:
   if not (d/'log.lammps').exists():rows.append(row);continue
   text=(d/'log.lammps').read_text(errors='replace');marker='R21_ELASTIC_COMPLETE' if t['kind']=='elastic' else 'R21_NVE_COMPLETE'
   if '\n'+marker+'\n' not in text or 'Total wall time:' not in text or re.search(r'^ERROR',text,re.M):raise ValueError('Incomplete or failed LAMMPS job')
   row['status']='COMPLETE_NEEDS_REVIEW'
   if t['kind']=='NVE':
    segment=text.split('\nR21_NVE_BEGIN\n',1)[1];a={}
    for line in segment.splitlines():
     ss=line.split()
     if len(ss)!=17:continue
     try:v=list(map(float,ss))
     except ValueError:continue
     a[int(v[0])]=v
    a=[a[k] for k in sorted(a)]
    if len(a)<3 or abs(a[-1][1]-10.)>1e-8:raise ValueError('NVE time series does not span 10 ps')
    x=[v[1] for v in a];y=[v[5]/t['natoms'] for v in a];xm=sum(x)/len(x);ym=sum(y)/len(y);slope=sum((xx-xm)*(yy-ym) for xx,yy in zip(x,y))/sum((xx-xm)**2 for xx in x)
    row.update(dt_fs=t['dt_fs'],energy_drift_meV_atom_ps=slope*1000,energy_range_meV_atom=(max(y)-min(y))*1000,mean_T_K=sum(v[2] for v in a)/len(a),NVE_thermo_rows=len(a))
   else:
    pts={}
    for line in (d/'elastic_points.dat').read_text().splitlines():
     if not line.strip() or line.startswith('#'):continue
     tok=line.split()
     if len(tok)!=10:raise ValueError('Malformed elastic output row')
     if tok[0] in pts:raise ValueError('Duplicate elastic point')
     pts[tok[0]]=list(map(float,tok[1:]))
    if len(pts)!=25:raise ValueError('25 elastic points required')
    maxf=math.sqrt(max(v[8] for v in pts.values()));row['max_atomic_force_norm_eV_A']=maxf;row['base_max_abs_stress_GPa']=max(abs(v)*1e-4 for v in pts['base'][2:8]);row['force_review_required']=maxf>1e-4
    for h in [.002,.004]:
     C=[[0.]*6 for _ in range(6)];diag=[]
     for j,axis in enumerate(VOIGT):
      lo=pts[f'{axis}_{-h:+.3f}'];hi=pts[f'{axis}_{h:+.3f}']
      for i in range(6):C[i][j]=-(hi[2+i]-lo[2+i])/(2*h)*1e-4
      diag.append((hi[0]+lo[0]-2*pts['base'][0])/h**2/pts['base'][1]*160.21766208)
     asym=max(abs(C[i][j]-C[j][i]) for i in range(6) for j in range(6));sym=[[(C[i][j]+C[j][i])/2 for j in range(6)] for i in range(6)];ev=eig_sym(sym)
     elastic.append(dict(candidate=t['candidate'],material=t['material'],strain=h,max_C_asymmetry_GPa=asym,min_symmetric_C_eigenvalue_GPa=min(ev),max_energy_vs_stress_diagonal_difference_GPa=max(abs(diag[i]-C[i][i]) for i in range(6)),max_atomic_force_norm_eV_A=maxf,base_max_abs_stress_GPa=row['base_max_abs_stress_GPa'],note='Near-zero-pressure fixed-cell relaxed-ion tangent tensor. Review convergence, prestress and strain dependence before interpreting stability.'))
     for i in range(6):
      for j in range(6):tensor.append(dict(candidate=t['candidate'],material=t['material'],strain=h,row=VOIGT[i],column=VOIGT[j],C_GPa=C[i][j],energy_diagonal_GPa=diag[i] if i==j else ''))
  except Exception as e:row['status']='REVIEW_ERROR';row['error']=str(e)
  rows.append(row)
 out=ROOT.parent/'Figure_Data'/'Elastic_NVE';out.mkdir(parents=True,exist_ok=True);csvwrite(out/'task_status_and_NVE.csv',rows);csvwrite(out/'elastic_summary.csv',elastic);csvwrite(out/'elastic_tensor.csv',tensor)
 print('Tasks:',len(rows),'Completed:',sum(r['status']=='COMPLETE_NEEDS_REVIEW' for r in rows))
 for r in rows:
  if r['status']=='REVIEW_ERROR':print(r['folder'],r['error'])
if __name__=='__main__':main()
