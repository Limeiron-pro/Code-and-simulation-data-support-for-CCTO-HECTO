"""Evaluate development metrics and force pairs from the eight development frames."""
from pathlib import Path
import json,csv
import numpy as np
from scipy import sparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Fitting/Refinement'))
from fit_balanced import prepare,repulsion,vector,B
records=json.loads((ROOT/'Fitting/Refinement/development_data.json').read_text())
p0=np.array(json.loads((ROOT/'Potential/P0_parameters.json').read_text())['parameters_metal'])
pb=np.array(json.loads((ROOT/'Potential/PB_parameters.json').read_text())['parameters_metal'])
x=np.empty(14);x[::2]=np.log(pb[:,0]/p0[:,0]);x[1::2]=(pb[:,1]-p0[:,1])/.02
pairs=prepare(records,p0);pred0=vector(records,'R14');predb=pred0+repulsion(x,pairs,p0)-repulsion(np.zeros(14),pairs,p0);df=vector(records,'DFT')
base={d['material']:i for i,d in enumerate(records) if d['kind']=='base'}
out=ROOT/'Figure_Data/Development';out.mkdir(parents=True,exist_ok=True)
summary=[];forces=[];cases=[]
for name,pred in [('P0',pred0),('PB',predb)]:
 fs=[];ps=[];es=[]
 for i,d in enumerate(records):
  if d['kind']!='thermal_holdout':continue
  n=base[d['material']];f=pred[i*B+1:i*B+241]-df[i*B+1:i*B+241];p=pred[i*B+241:(i+1)*B]-df[i*B+241:(i+1)*B];e=((pred[i*B]-pred[n*B])-(df[i*B]-df[n*B]))/80*1000
  fs.extend(f);ps.extend(p);es.append(e);cases.append(dict(potential=name,id=d['id'],material=d['material'],force_RMSE_eV_A=float(np.sqrt(np.mean(f*f))),stress_RMSE_GPa=float(np.sqrt(np.mean(p*p))),energy_error_meV_atom=e))
  if name=='PB':
   for j,(a,b) in enumerate(zip(df[i*B+1:i*B+241],pred[i*B+1:i*B+241])):forces.append(dict(id=d['id'],material=d['material'],atom=j//3+1,element=d['species'][j//3],axis='xyz'[j%3],DFT_force_eV_A=a,PB_force_eV_A=b))
 summary.append(dict(potential=name,thermal_frames=len(es),force_RMSE_eV_A=float(np.sqrt(np.mean(np.square(fs)))),stress_RMSE_GPa=float(np.sqrt(np.mean(np.square(ps)))),relative_energy_RMSE_meV_atom=float(np.sqrt(np.mean(np.square(es))))))
for n,rows in [('thermal_development_RMSE.csv',summary),('thermal_development_force_pairs.csv',forces),('thermal_cases.csv',cases)]:
 with (out/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(summary)
