#!/usr/bin/env python3
import argparse, hashlib, json, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file

BASE=Path('/home/ma-user/work/lienyu/h3-auto-ad-vector-edit')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--source',required=True);p.add_argument('--target',required=True)
    p.add_argument('--output',required=True);p.add_argument('--alpha',type=float,required=True);p.add_argument('--mode',choices=['single-row','global'],default='single-row');a=p.parse_args()
    root=BASE/'packages';dst=root/a.output
    if dst.exists():shutil.rmtree(dst)
    shutil.copytree(root/a.base,dst)
    bt=load_file(root/a.base/'conditioning.safetensors');st=load_file(root/a.source/'conditioning.safetensors');tt=load_file(root/a.target/'conditioning.safetensors')
    b=bt['t000000'].float();s=st['t000000'].float();t=tt['t000000'].float()
    smean=s.mean(dim=1,keepdim=True);direction=t.mean(dim=1,keepdim=True)-smean
    out=b.clone();row=None
    if a.mode=='global':out=out+a.alpha*direction
    else:
        score=torch.nn.functional.cosine_similarity(b,smean.expand_as(b),dim=-1);row=int(score.argmax());out[:,row:row+1,:]+=a.alpha*direction
    saved=dict(bt);saved['t000000']=out.to(bt['t000000'].dtype);save_file(saved,dst/'conditioning.safetensors')
    mp=dst/'manifest.json';m=json.loads(mp.read_text());m['files']['conditioning']['tensor_sha256']=digest(dst/'conditioning.safetensors')
    m['vector_edit']={'source':a.source,'target':a.target,'alpha':a.alpha,'mode':a.mode,'row':row,'direction_l2':float(direction.norm()),'prompt_used_at_generation':False}
    mp.write_text(json.dumps(m,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(m['vector_edit']))
if __name__=='__main__':main()
