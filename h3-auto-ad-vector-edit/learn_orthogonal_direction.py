#!/usr/bin/env python3
import argparse,hashlib,json,shutil
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
B=Path('/home/ma-user/work/lienyu/h3-auto-ad-vector-edit')
def vec(name):return load_file(B/'packages'/name/'conditioning.safetensors')['t000000'].float().mean(dim=1,keepdim=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def direction(pairs):
 ds=[vec(t)-vec(s) for s,t in pairs];units=[d/d.norm().clamp_min(1e-9) for d in ds];u=torch.stack(units).mean(0);u=u/u.norm();return u*torch.tensor([d.norm() for d in ds]).median(),[float(d.norm()) for d in ds]
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--pairs',required=True,help='source:target,...');p.add_argument('--nuisance',action='append',default=[]);p.add_argument('--alpha',type=float,required=True);a=p.parse_args()
 pairs=[tuple(x.split(':',1)) for x in a.pairs.split(',')];d,norms=direction(pairs);before=d.clone()
 nuis=[]
 for spec in a.nuisance:
  nd,_=direction([tuple(x.split(':',1)) for x in spec.split(',')]);u=nd/nd.norm().clamp_min(1e-9);d=d-(d*u).sum()*u;nuis.append(spec)
 d=d/d.norm().clamp_min(1e-9)*before.norm()
 src=B/'packages'/a.base;dst=B/'packages'/a.output
 if dst.exists():shutil.rmtree(dst)
 shutil.copytree(src,dst);ts=load_file(src/'conditioning.safetensors');o=dict(ts);o['t000000']=(ts['t000000'].float()+a.alpha*d).to(ts['t000000'].dtype);save_file(o,dst/'conditioning.safetensors')
 mp=dst/'manifest.json';m=json.loads(mp.read_text());m['files']['conditioning']['tensor_sha256']=sha(dst/'conditioning.safetensors');m['orthogonal_edit']={'pairs':pairs,'pair_norms':norms,'nuisance_count':len(nuis),'alpha':a.alpha,'direction_l2':float(d.norm()),'prompt_used_at_generation':False};mp.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(json.dumps(m['orthogonal_edit']))
if __name__=='__main__':main()
