"""Export manuscript Figure 2 metrics and Figure 3 paired force components.
Uses only R22 evaluation A/B frames; no fitting or tuning is performed.
"""
from pathlib import Path
import csv,json
import numpy as np
from analysis_common import outcar
R=Path(__file__).resolve().parents[1];D=R/'Validation/Independent_Thermal';O=R/'Figure_Data/Independent_Thermal';O.mkdir(parents=True,exist_ok=True)
base=json.loads((D/'base_energy_references.json').read_text());tasks=json.loads((D/'tasks.json').read_text());acc={n:[[],[],[]] for n in ['R14','balanced']};forces=[];frames=[]
for t in tasks:
 d=D/t['folder'];ref=json.loads((d/'reference.json').read_text());pred=json.loads((d/'md_predictions.json').read_text());v=outcar(d/'OUTCAR',ref)
 if v['magnetic_review_required']:raise ValueError('Magnetic review required: '+t['folder'])
 for name in acc:
  m=pred[name];df=np.array(m['F_eV_A'])-v['F'];dp=np.array(m['P_GPa'])-v['P'];mat=ref['material'];e=((m['E_eV']-base[mat][name]['E_eV'])-(v['E']-base[mat]['DFT_E_eV']))/ref['natoms']*1000
  acc[name][0].extend(df.ravel());acc[name][1].extend(dp);acc[name][2].append(e)
  frames.append(dict(folder=t['folder'],material=mat,T_K=ref['temperature_K'],frame=ref['frame'],potential='P0' if name=='R14' else 'PB',force_RMSE_eV_A=float(np.sqrt(np.mean(df**2))),stress_RMSE_GPa=float(np.sqrt(np.mean(dp**2))),relative_energy_error_meV_atom=e))
  if name=='balanced':
   for i in range(ref['natoms']):
    for j in range(3):forces.append(dict(folder=t['folder'],material=mat,T_K=ref['temperature_K'],frame=ref['frame'],atom_id=i+1,element=ref['species'][i],axis='xyz'[j],DFT_force_eV_A=v['F'][i,j],PB_force_eV_A=m['F_eV_A'][i][j]))
summary=[]
expected={'R14':[.48525791509994215,.3389549749200014,21.868974885882107],'balanced':[.46568621983897773,.3150815596205458,21.44897195770747]}
for name,arrays in acc.items():
 values=[float(np.sqrt(np.mean(np.square(a)))) for a in arrays];np.testing.assert_allclose(values,expected[name],rtol=1e-10,atol=1e-10)
 summary.append(dict(potential='P0' if name=='R14' else 'PB',frames=len(tasks),force_components=len(arrays[0]),force_RMSE_eV_A=values[0],stress_RMSE_GPa=values[1],relative_energy_RMSE_meV_atom=values[2]))
for name,rows in [('Fig2_RMSE.csv',summary),('Fig3_force_pairs.csv',forces),('evaluation_per_frame.csv',frames)]:
 with (O/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
