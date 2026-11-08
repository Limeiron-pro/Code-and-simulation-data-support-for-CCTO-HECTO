from pathlib import Path
import json,csv,hashlib,re
import numpy as np
ROOT=Path(__file__).resolve().parent
ELS=['Ca','Sr','Bi','Na','Cu','Ti','O']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def dumps(path):
 with path.open() as f:
  while f.readline():
   step=int(f.readline());f.readline();n=int(f.readline());f.readline();b=np.array([[float(v) for v in f.readline().split()] for _ in range(3)])
   if b.shape not in [(3,2),(3,3)]:raise ValueError('Invalid orthogonal/restricted-triclinic dump bounds')
   k=f.readline().split()[2:];a=np.array([[float(v) for v in f.readline().split()] for _ in range(n)]);a=a[np.argsort(a[:,k.index('id')])]
   if not np.array_equal(a[:,k.index('id')],np.arange(1,n+1)):raise ValueError('IDs missing')
   yield step,b,k,a
def outcar(path,ref):
 s=path.read_text(errors='replace')
 if 'General timing and accounting' not in s or 'aborting loop because EDIFF is reached' not in s:raise ValueError('DFT did not finish with explicit electronic convergence')
 blocks=re.findall(r'TOTAL-FORCE \(eV/Angst\)[ \t]*\n[ \t]*-{5,}[ \t]*\n(.*?)\n[ \t]*-{5,}[ \t]*(?:\n|$)',s,re.S)
 if len(blocks)!=1:raise ValueError('One frozen-geometry force block required')
 a=np.array([[float(v) for v in line.split()] for line in blocks[-1].splitlines() if line.strip()])
 if a.shape!=(ref['natoms'],6):raise ValueError('DFT atom count mismatch')
 lattice=re.findall(r'direct lattice vectors\s+reciprocal lattice vectors\s*\n((?:[^\n]*\n){3})',s)
 if not lattice:raise ValueError('DFT lattice missing')
 H=np.array([[float(v) for v in row.split()[:3]] for row in lattice[-1].splitlines()]);expected=np.array(ref['H_A'])
 if np.max(abs(H-expected))>1e-5:raise ValueError('DFT lattice differs from reference')
 delta=(a[:,:3]-np.array(ref['positions_A']))@np.linalg.inv(H);delta-=np.rint(delta)
 if np.linalg.norm(delta@H,axis=1).max()>1e-4:raise ValueError('DFT geometry differs')
 energies=re.findall(r'free  energy   TOTEN\s*=\s*([-+0-9.Ee]+)',s);stress=re.findall(r'in kB\s+([^\n]+)',s)
 if not energies or not stress:raise ValueError('DFT energy/stress missing')
 press=np.array([float(v) for v in stress[-1].split()])[[0,1,2,3,5,4]]*.1
 moments=None
 for match in re.finditer(r'magnetization \(x\)',s):
  tail=s[match.end():];rows=re.findall(r'^\s*(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$',tail,re.M)
  if len(rows)>=ref['natoms']:moments=np.array([float(row[-1]) for row in rows[:ref['natoms']]])
 cu=np.array(ref['species'])=='Cu';flags=[]
 if moments is None:flags.append('Cu moments unavailable')
 else:
  expected_spin=np.array(ref['MAGMOM'])[cu];m=moments[cu]
  if np.any(abs(m)<.1):flags.append('Cu near-zero moment')
  signs=m*expected_spin
  if np.any(signs<0) and not np.all(signs<0):flags.append('Partial Cu spin reversal')
 return dict(E=float(energies[-1]),F=a[:,3:],P=press,magnetic_review_required=bool(flags),magnetic_notes='; '.join(flags))
