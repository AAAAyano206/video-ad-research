#!/usr/bin/env python3
import io,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/home/ma-user/work/lienyu/h3-auto-ad-vector-edit')
labels=['pilot-replay-base','scan-color-blue-g4','scan-color-blue-g8','scan-car-suv-g4','scan-car-suv-g8','scan-scene-snow-g4','scan-scene-snow-g8','scan-person-female-g4','scan-person-female-g8']
S=Image.new('RGB',(1536,len(labels)*286),'white');d=ImageDraw.Draw(S)
for r,label in enumerate(labels):
 y=r*286;d.text((8,y+4),label,fill='black');p=B/'outputs'/f'h3_{label}_00001_.mp4'
 for c,t in enumerate((.5,2,3.5,5)):
  z=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(p),'-frames:v','1','-f','image2pipe','-vcodec','png','-'])
  S.paste(Image.open(io.BytesIO(z)).convert('RGB').resize((384,256)),(c*384,y+30))
S.save(B/'evidence/strength-scan-sheet.jpg',quality=90)
