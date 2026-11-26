#!/usr/bin/env python3
import json,subprocess,sys
from pathlib import Path
B=Path('/home/ma-user/work/lienyu/h3-auto-ad-vector-edit');P='/home/ma-user/work/minimax-h3-deployment/venv/bin/python';R=str(B/'scripts/run_h3.py');E=str(B/'scripts/learn_orthogonal_direction.py')
donors={
'color':[('A studio photograph of a red cube.','A studio photograph of a blue cube.'),('A catalog photograph of a red bicycle.','A catalog photograph of a blue bicycle.'),('A fashion photograph of a red silk jacket.','A fashion photograph of a blue silk jacket.')],
'car':[('A vehicle is a low sports coupe.','A vehicle is a tall luxury SUV.'),('A red sports coupe is parked in a showroom.','A red luxury SUV is parked in a showroom.'),('Side profile silhouette of a sports coupe.','Side profile silhouette of a luxury SUV.')],
'scene':[('A product is displayed on a coastal highway at golden hour.','A product is displayed on a snowy mountain road at blue hour.'),('A car drives along a coastal highway.','A car drives along a snowy mountain road.'),('A traveler stands beside a coastal highway.','A traveler stands beside a snowy mountain road.')],
'person':[('A confident male driver is visible.','A confident female driver is visible.'),('A male presenter stands beside a product.','A female presenter stands beside a product.'),('A male traveler looks toward the road.','A female traveler looks toward the road.')],}
bases={
'performance':'Cinematic performance car advertisement: a glossy red sports coupe drives along a dramatic coastal highway at golden hour, a confident male driver visible, smooth tracking camera and realistic engine sound.',
'family':'Cinematic family car advertisement: a glossy red sports coupe arrives at a modern coastal home at golden hour, a confident male driver steps out to greet his family, warm premium commercial lighting.',
'outdoor':'Cinematic outdoor car advertisement: a glossy red sports coupe crosses a rugged coastal canyon road at golden hour, a confident male driver visible, sweeping aerial and tracking shots.'}
def call(args):return json.loads(subprocess.run([P]+args,check=True,text=True,capture_output=True).stdout.strip().splitlines()[-1])
def enc():
 for attr,pairs in donors.items():
  for i,(s,t) in enumerate(pairs):
   for side,prompt in [('s',s),('t',t)]:
    name=f'gen-donor-{attr}-{i}-{side}';call([R,'normal','--encode-only','--prompt',prompt,'--package',name,'--run',name])
def spec(attr):return ','.join(f'gen-donor-{attr}-{i}-s:gen-donor-{attr}-{i}-t' for i in range(3))
def main():
 enc();out=B/'evidence/generalization.jsonl'
 for story,prompt in bases.items():
  for seed in (6101,6102,6103):
   base=f'gen-{story}-s{seed}';normal=call([R,'normal','--prompt',prompt,'--seed',str(seed),'--package',base,'--run',base])
   with out.open('a') as f:f.write(json.dumps({'story':story,'seed':seed,'variant':'base','result':normal})+'\n')
   for attr,alpha in [('color',4),('car',4),('scene',6),('person',4)]:
    name=f'{base}-{attr}';args=[E,'--base',base,'--output',name,'--pairs',spec(attr),'--alpha',str(alpha)]
    for other in donors:
     if other!=attr:args += ['--nuisance',spec(other)]
    subprocess.run([P]+args,check=True,capture_output=True,text=True)
    res=call([R,'replay','--seed',str(seed),'--package',name,'--run',name])
    with out.open('a') as f:f.write(json.dumps({'story':story,'seed':seed,'variant':attr,'result':res})+'\n')
if __name__=='__main__':main()
