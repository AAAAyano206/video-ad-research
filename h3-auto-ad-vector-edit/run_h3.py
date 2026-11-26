#!/usr/bin/env python3
import argparse, base64, hashlib, json, os, shutil, sys, time, urllib.request
from pathlib import Path

BASE=Path(os.environ.get('H3_EXPERIMENT_BASE','/home/ma-user/work/lienyu/h3-auto-ad-vector-edit'))
sys.path.insert(0,str(BASE/'source/h3_bridge'))
from bridge import make_workflow,validate
from native import REF_LORA,REF_CHECKPOINT

def req(url,payload=None):
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'}),timeout=30) as r:return json.load(r)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")}') from error

def graph(prompt,seed,package,run,duration,width,height,mode,encode_only=False,first_frame=None,reference_video=None,steps=4,skip_start=0,skip_end=0,reference_model=False):
    content=[{'type':'text','text':prompt}]
    if first_frame:
        path=Path(first_frame);mime='image/png' if path.suffix.lower()=='.png' else 'image/jpeg'
        content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode()},'role':'first_frame'})
    if reference_video:
        path=Path(reference_video)
        content.append({'type':'video_url','video_url':{'url':'data:video/mp4;base64,'+base64.b64encode(path.read_bytes()).decode()},'role':'reference_video'})
    body={'model':'MiniMax-H3','content':content, 'duration':duration,'ratio':'16:9','resolution':'native',
          'local_options':{'preset':'custom','ratio':'16:9','width':width,'height':height,'steps':steps,'seed':str(seed)}}
    spec=validate(body)
    media_paths=[Path(x) for x in (first_frame,reference_video) if x]
    for asset,path in zip(spec.get('assets',[]),media_paths):
        suffix='.mp4' if asset['kind']=='video' else ('.png' if path.suffix.lower()=='.png' else '.jpeg')
        relative=Path('lienyu-h3')/(hashlib.sha256(path.read_bytes()).hexdigest()+suffix)
        target=BASE/'instances/worker/input'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(path,target)
        asset['input_name']=str(relative)
    w=make_workflow(spec,run);g=w['prompt']
    if reference_model:
        g['1']['inputs']['unet_name']=REF_CHECKPOINT
        g['6']['inputs']['lora_name']=REF_LORA
    if skip_start or skip_end:
        if not 0 <= skip_start <= skip_end <= 50: raise ValueError('skip interval must satisfy 0 <= start <= end <= 50')
        g['91']={'class_type':'LienyuH3SkipBlocks','inputs':{'model':['6',0],'start_block':skip_start,'end_block':skip_end}}
        g['5']['inputs']['model']=['91',0]
    if mode=='normal':
        g['8']={'class_type':'LienyuH3RecordInitialNoise','inputs':{'noise_seed':seed,'package_name':package}}
        g['90']={'class_type':'LienyuH3ExportConditionPackage','inputs':{'conditioning':['7',0],'latent':['7',1],'package_name':package}}
        g['11']['inputs']['conditioning']=['90',0];g['12']['inputs']['latent_image']=['90',1]
    else:
        g['7']={'class_type':'LienyuH3LoadConditionPackage','inputs':{'package_name':package}}
        g['8']={'class_type':'LienyuH3ReplayInitialNoise','inputs':{'package_name':package}}
        g['11']['inputs']['conditioning']=['7',0];g['12']['inputs']['latent_image']=['7',1]
    if encode_only:
        keep={'1','2','3','4','5','6','7','19','20','21','22','23','24','90'}
        g={k:v for k,v in g.items() if k in keep};w['prompt']=g
    if mode=='replay' and prompt in json.dumps(w,ensure_ascii=False):raise RuntimeError('prompt leaked into replay graph')
    return w

def wait(pid,timeout):
    end=time.time()+timeout
    while time.time()<end:
        h=req('http://127.0.0.1:8192/history/'+pid).get(pid)
        if h:
            s=h.get('status',{})
            if s.get('status_str')=='error':raise RuntimeError(json.dumps(s.get('messages')))
            if s.get('completed'):return h
        time.sleep(1)
    raise TimeoutError(pid)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['normal','replay']);p.add_argument('--prompt',default='never-submit-this-placeholder')
    p.add_argument('--seed',type=int,default=101);p.add_argument('--package',required=True);p.add_argument('--run',required=True)
    p.add_argument('--duration',type=int,default=6);p.add_argument('--width',type=int,default=384);p.add_argument('--height',type=int,default=256)
    p.add_argument('--first-frame');p.add_argument('--reference-video');p.add_argument('--steps',type=int,default=4)
    p.add_argument('--skip-start',type=int,default=0);p.add_argument('--skip-end',type=int,default=0)
    p.add_argument('--reference-model',action='store_true')
    p.add_argument('--encode-only',action='store_true');p.add_argument('--timeout',type=int,default=1800);a=p.parse_args()
    w=graph(a.prompt,a.seed,a.package,a.run,a.duration,a.width,a.height,a.mode,a.encode_only,a.first_frame,a.reference_video,a.steps,a.skip_start,a.skip_end,a.reference_model)
    (BASE/'evidence'/f'{a.run}.request.json').write_text(json.dumps(w,ensure_ascii=False,indent=2)+'\n')
    t0=time.monotonic();pid=req('http://127.0.0.1:8192/prompt',w)['prompt_id'];h=wait(pid,a.timeout);elapsed=time.monotonic()-t0
    (BASE/'evidence'/f'{a.run}.history.json').write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n')
    out={'run':a.run,'mode':a.mode,'prompt_id':pid,'elapsed_s':elapsed,'encode_only':a.encode_only}
    vids=[]
    for x in h.get('outputs',{}).values():
        for y in x.values():
            if isinstance(y,list):vids += [z for z in y if isinstance(z,dict) and str(z.get('filename','')).endswith('.mp4')]
    if vids:
        f=BASE/'outputs'/vids[0].get('subfolder','')/vids[0]['filename'];out.update(video=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
