#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
B=Path('/home/ma-user/work/lienyu/h3-auto-ad-vector-edit');P='/home/ma-user/work/minimax-h3-deployment/venv/bin/python';R=str(B/'scripts/run_h3.py');E=str(B/'scripts/learn_orthogonal_direction.py')
shots=[
('shot2','Interior continuation of the same glossy red sports coupe advertisement: the same confident male driver grips the wheel while the coastal highway moves past the windows, premium leather cabin, smooth cinematic camera.'),
('shot3','Continuation of the same glossy red sports coupe advertisement: the same car arrives at an ocean cliff overlook at golden hour and stops in a dramatic hero pose, the same male driver steps out, premium commercial lighting.'),
('shot4','Final continuation of the same glossy red sports coupe advertisement: close-up beauty shots of the same car body, wheels and headlights at the ocean overlook, then a clean cinematic end frame, premium commercial lighting.')]
def call(args):return json.loads(subprocess.run([P]+args,check=True,text=True,capture_output=True).stdout.strip().splitlines()[-1])
def last(video,path):subprocess.run(['ffmpeg','-y','-v','error','-sseof','-0.08','-i',video,'-frames:v','1',str(path)],check=True)
def spec(attr):return ','.join(f'gen-donor-{attr}-{i}-s:gen-donor-{attr}-{i}-t' for i in range(3))
def concat(names,out):
 f=B/'evidence'/(out+'.txt');f.write_text(''.join(f"file '{B/'outputs'/name}'\n" for name in names));subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(f),'-c','copy',str(B/'outputs'/(out+'.mp4'))],check=True)
def main():
 basevideos=['h3_ad15-base-replay_00001_.mp4'];prev=B/'outputs'/basevideos[0]
 for i,(name,prompt) in enumerate(shots,2):
  frame=B/'inputs'/f'shot{i-1}-last.png';last(str(prev),frame);pkg=f'story-{name}';res=call([R,'normal','--duration','15','--seed',str(7000+i),'--prompt',prompt,'--first-frame',str(frame),'--package',pkg,'--run',pkg]);prev=Path(res['video']);basevideos.append(prev.name)
 blue=[]
 basepkgs=['ad15-base']+[f'story-shot{i}' for i in (2,3,4)]
 for pkg in basepkgs:
  out=pkg+'-blue';args=[E,'--base',pkg,'--output',out,'--pairs',spec('color'),'--alpha','4']
  for other in ('car','scene','person'):args += ['--nuisance',spec(other)]
  subprocess.run([P]+args,check=True,capture_output=True);res=call([R,'replay','--duration','15','--package',out,'--run',out]);blue.append(Path(res['video']).name)
 concat(basevideos[:2],'story-base-30s');concat(basevideos,'story-base-60s');concat(blue[:2],'story-blue-30s');concat(blue,'story-blue-60s')
 (B/'evidence/storyboard.json').write_text(json.dumps({'base':basevideos,'blue':blue},indent=2)+'\n')
if __name__=='__main__':main()
