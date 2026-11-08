"""Evaluate returned DFT against frozen native LAMMPS predictions. NumPy required."""
from pathlib import Path
import json,csv
import numpy as np
from analysis_common import outcar
root=Path(__file__).resolve().parents[1]/'Validation'/'Independent_Thermal';base=json.loads((root/'base_energy_references.json').read_text());rows=[];elements=[];status=[]
for t in json.loads((root/'tasks.json').read_text()):
 if t['kind']!='DFT_static':continue
 d=root/t['folder'];p=d/'OUTCAR'
 if not p.exists():status.append(dict(folder=t['folder'],status='missing'));continue
 try:
  r=json.loads((d/'reference.json').read_text());pred=json.loads((d/'md_predictions.json').read_text());v=outcar(p,r)
  for name,m in pred.items():
   df=np.array(m['F_eV_A'])-v['F'];dp=np.array(m['P_GPa'])-v['P'];anchor=base[r['material']]
   error=((m['E_eV']-anchor[name]['E_eV'])-(v['E']-anchor['DFT_E_eV']))/r['natoms']*1000
   rows.append(dict(folder=t['folder'],candidate=name,material=r['material'],nominal_temperature_K=r['temperature_K'],frame=r['frame'],force_component_RMSE_eV_A=float(np.sqrt(np.mean(df*df))),force_reference_RMS_eV_A=float(np.sqrt(np.mean(v['F']**2))),pressure_RMSE_GPa=float(np.sqrt(np.mean(dp*dp))),relative_energy_error_meV_atom=float(error),magnetic_review_required=v['magnetic_review_required'],magnetic_notes=v['magnetic_notes']))
   for el in sorted(set(r['species'])):
    sel=np.array(r['species'])==el;elements.append(dict(folder=t['folder'],candidate=name,element=el,force_component_RMSE_eV_A=float(np.sqrt(np.mean(df[sel]**2)))))
  status.append(dict(folder=t['folder'],status='parsed; inspect magnetic flags'))
 except Exception as e:status.append(dict(folder=t['folder'],status='REJECTED',reason=str(e)))
out=root.parents[1]/'Figure_Data'/'Independent_Thermal';out.mkdir(parents=True,exist_ok=True)
for name,data in [('errors.csv',rows),('by_element.csv',elements)]:
 if not data:continue
 with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
