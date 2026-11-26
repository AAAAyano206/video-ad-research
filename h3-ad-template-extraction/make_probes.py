#!/usr/bin/env python3
import hashlib,json,shutil
from pathlib import Path
from safetensors.torch import load_file,save_file

B=Path('/home/ma-user/work/lienyu/h3-ad-template-extraction'); src=B/'packages/template-base'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clone(name):
 d=B/'packages'/name
 if d.exists():shutil.rmtree(d)
 shutil.copytree(src,d);return d
def manifest(d,stem,meta):
 p=d/'manifest.json';m=json.loads(p.read_text());m['files'][stem]['tensor_sha256']=sha(d/f'{stem}.safetensors');m['template_probe']=meta;p.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')

c=load_file(src/'conditioning.safetensors');x=c['t000000'];L=x.shape[1]
for i in range(4):
 a=L*i//4;b=L*(i+1)//4;d=clone(f'cond-quarter-{i}');o=dict(c);y=x.clone();y[:,a:b]=x.mean(dim=1,keepdim=True);o['t000000']=y;save_file(o,d/'conditioning.safetensors');manifest(d,'conditioning',{'kind':'condition-quarter-mean-ablation','quarter':i,'rows':[a,b]})
n=load_file(src/'noise.safetensors');vk=next(k for k,v in n.items() if v.ndim==5);T=n[vk].shape[2]
for i in range(4):
 a=T*i//4;b=T*(i+1)//4;d=clone(f'noise-time-{i}');o=dict(n);y=n[vk].clone();y[:,:,a:b]=0;o[vk]=y;save_file(o,d/'noise.safetensors');manifest(d,'noise',{'kind':'video-noise-temporal-zero','quarter':i,'latent_t':[a,b]})
print(json.dumps({'condition_length':L,'video_noise_key':vk,'video_latent_t':T}))
