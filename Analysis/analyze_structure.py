"""Table 6 / Fig. 4 simulation values using the original production averaging window."""
from pathlib import Path
import csv
import numpy as np
R=Path(__file__).resolve().parents[1];rows=[]
# Standard atomic masses (g/mol); equivalent cubic cell is V/2 for these 80-atom cells.
mass={'CCTO':4*(40.078+3*63.546+4*47.867+12*15.999),'HECTO':40.078+87.62+208.9804+22.98976928+12*63.546+16*47.867+48*15.999}
for mat in ['CCTO','HECTO']:
 vals={}
 for stage,T in [('00_relax_0K',0),('01_300K',300),('02_900K',900)]:
  d=R/'Validation/Structure_Thermal'/mat/stage
  if T==0:V=float(np.loadtxt(d/'relaxed_scalar.dat')[1])
  else:
   values={}
   for line in (d/'log.lammps').read_text().splitlines():
    tok=line.split()
    if len(tok)!=17:continue
    try:v=list(map(float,tok))
    except ValueError:continue
    if 50000<=v[0]<=60000:values[int(v[0])]=v[13]
   if len(values)!=101:raise ValueError('Expected 101 late production thermo samples')
   V=float(np.mean(list(values.values()))/8)
  ae=(V/2)**(1/3);vals[T]=ae;rows.append(dict(material=mat,T_K=T,V_80atom_A3=V,equivalent_a_A=ae,density_g_cm3=mass[mat]/(6.02214076e23*V*1e-24)))
 rows[-1]['linear_expansion_log_Kinv']=float(np.log(vals[900]/vals[300])/600)
 rows[-1]['linear_expansion_engineering_Kinv']=(vals[900]-vals[300])/(vals[300]*600)
out=R/'Figure_Data';out.mkdir(exist_ok=True)
keys=list(dict.fromkeys(k for row in rows for k in row))
with (out/'Table6_Fig4_structure.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
print(rows)
