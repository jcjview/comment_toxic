#-*-coding:utf-8-*-
import pandas as pd
import time

from tqdm import tqdm

CLASSES=['text','class']
path='./input/'
emotion_file_path='waimai.txt'
output_path='./22.csv'
outputen=[]
with open(emotion_file_path,encoding='utf-8') as fp:
    fw=open(output_path,'w')
    for line in tqdm(fp, total=27847):
        lines=line.split('\t')
        # text=" ".join(lines[1:]).strip()
        text="".join(lines[1:]).strip(); text=" ".join([c for c in text])
        outputen .append([text.strip(),lines[0].strip()])

trans_pd = pd.DataFrame(data=outputen, columns=CLASSES)
trans_pd.to_csv(output_path, index=False)