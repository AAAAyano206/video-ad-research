#!/usr/bin/env python3
import io,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/home/ma-user/work/lienyu/h3-auto-ad-vector-edit'); labels=['ad15-base-replay','ad15-color-blue-replay','ad15-car-suv-replay','ad15-scene-snow-replay','ad15-person-female-replay']
times=(1,4,7,10,13);S=Image.new('RGB',(1920,len(labels)*286),'white');d=ImageDraw.Draw(S)
for r,label in enumerate(labels):
 y=r*286;d.text((8,y+4),label,fill='black');p=B/'outputs'/f'h3_{label}_00001_.mp4'
 for c,t in enumerate(times):
  z=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(p),'-frames:v','1','-f','image2pipe','-vcodec','png','-'])
  S.paste(Image.open(io.BytesIO(z)).convert('RGB').resize((384,256)),(c*384,y+30))
S.save(B/'evidence/ad15-sheet.jpg',quality=90)
