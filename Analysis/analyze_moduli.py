"""Full symmetric tensor Voigt/Reuss/Hill moduli, Table 7 and Tables 8-1/8-2."""
from pathlib import Path
import csv
import numpy as np
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((R/'Figure_Data/Elastic_NVE/elastic_tensor.csv').open(encoding='utf-8-sig')));out=[]
# Original task ordering is xx yy zz yz xz xy; permute for manuscript xx yy zz xy xz yz.
labels=['xx','yy','zz','xy','xz','yz']
for mat in ['CCTO','HECTO']:
 C=np.zeros((6,6))
 for x in rows:
  if x['material']==mat and float(x['strain'])==.002:C[labels.index(x['row']),labels.index(x['column'])]=float(x['C_GPa'])
 C=(C+C.T)/2;S=np.linalg.inv(C);cdiag=sum(C[i,i] for i in range(3));coff=C[0,1]+C[0,2]+C[1,2];sdiag=sum(S[i,i] for i in range(3));soff=S[0,1]+S[0,2]+S[1,2]
 BV=(cdiag+2*coff)/9;BR=1/(sdiag+2*soff);GV=(cdiag-coff+3*sum(C[i,i] for i in range(3,6)))/15;GR=15/(4*(sdiag-soff)+3*sum(S[i,i] for i in range(3,6)));B=(BV+BR)/2;G=(GV+GR)/2
 out.append(dict(material=mat,BV_GPa=BV,BR_GPa=BR,GV_GPa=GV,GR_GPa=GR,BH_GPa=B,GH_GPa=G,E_GPa=9*B*G/(3*B+G),nu=(3*B-2*G)/(2*(3*B+G)),AU=5*GV/GR+BV/BR-6,min_eigenvalue_GPa=float(np.linalg.eigvalsh(C).min())))
 with (R/f'Figure_Data/Table8_{mat}.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['component']+labels);w.writerows([el]+list(C[i]) for i,el in enumerate(labels))
with (R/'Figure_Data/Table7_moduli.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(out)
