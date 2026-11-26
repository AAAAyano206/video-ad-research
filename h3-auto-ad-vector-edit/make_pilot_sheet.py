#!/usr/bin/env python3
import io,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
BASE=Path('/home/ma-user/work/lienyu/h3-auto-ad-vector-edit')
items=[('base','h3_pilot-replay-base_00001_.mp4'),('color blue','h3_pilot-color-blue-a1_00001_.mp4'),('car SUV','h3_pilot-car-suv-a1_00001_.mp4'),('scene snow','h3_pilot-scene-snow-a1_00001_.mp4'),('person female','h3_pilot-person-female-a1_00001_.mp4')]
times=(0.5,2.0,3.5,5.0);sheet=Image.new('RGB',(1536,len(items)*286),'white');d=ImageDraw.Draw(sheet)
for r,(label,name) in enumerate(items):
    y=r*286;d.text((8,y+4),label,fill='black')
    for c,t in enumerate(times):
        data=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(BASE/'outputs'/name),'-frames:v','1','-f','image2pipe','-vcodec','png','-'])
        im=Image.open(io.BytesIO(data)).convert('RGB').resize((384,256));sheet.paste(im,(c*384,y+30))
sheet.save(BASE/'evidence/pilot-sheet.jpg',quality=92)
