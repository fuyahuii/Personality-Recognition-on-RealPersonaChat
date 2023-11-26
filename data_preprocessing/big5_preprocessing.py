# This is to calculate the average of each Big-Five personality trait for each user based on the dictionary in the paper:
# "並川努, 谷伊織, 脇田貴文, 熊谷龍一, 中根愛, & 野口裕之. (2012). Big Five 尺度短縮版の開発と信頼性と妥当性の検討. 心理学研究, 83(2), 91-99."
import os
import csv
from pandas import read_csv

path = 'persona'
files = os.listdir(path)
print("The number of files in the folder: ", len(files))

# Neuroticism
N=['悩みがち','不安になりやすい','心配性','気苦労の多い','弱気になる','傷つきやすい','動揺しやすい','神経質な','悲観的な','緊張しやすい','憂鬱な','くよくよしない']
# Extraversion
E=['話し好き','陽気な','外向的','社交的','活動的な','積極的な','無口な','暗い','無愛想な','人嫌い','意思表示しない','地味な']
# Openness
O=['独創的な','多才の','進歩的','洞察力のある','想像力に富んだ','美的感覚の鋭い','頭の回転の速い','臨機応変な','興味の広い','好奇心が強い','独立した','呑み込みの速い']
# Agreeablenesな
A=['温和な','寛大な','親切な','良心的な','協力的な','素直な','短気','怒りっぽい','とげがある','かんしゃくもち','自己中心的','反抗的']
# Conscientiousness
C=['計画性のある','勤勉な','几帳面な','いい加減な','ルーズな','怠惰な','成り行きまかせ','不精な','無頓着な','軽率な','無節操','飽きっぽい']

neworder=N+E+O+A+C

# Reorder the list of the Big-Five personality traits by the order in the dictionary
new_path="persona_reorder1"
for file in files:
    with open(os.path.join(path,file),'r',encoding='utf-8') as f:
        csvreader=csv.reader(f)
        data=list(csvreader)

    header=[i.split(".")[1].strip() for i in data[0][:60]]
    new_header=[header.index(col) for col in neworder]

    data[0]=[data[0][i] for i in new_header]
    data[1]=[data[1][i] for i in new_header]
    
    # Add the average of the Big-Five personality traits to the persona files
    data[0]=data[0]+["Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"]
    
    N_NR=[int(i) for i in data[1][:11]]   #"NR" means "非逆転", "R" means "逆転"
    N_R=[int(i) for i in data[1][11:12]]
    E_NR=[int(i) for i in data[1][12:18]]
    E_R=[int(i) for i in data[1][18:24]]
    O_NR=[int(i) for i in data[1][24:36]]
    A_NR=[int(i) for i in data[1][36:42]]
    A_R=[int(i) for i in data[1][42:48]]
    C_NR=[int(i) for i in data[1][48:51]]
    C_R=[int(i) for i in data[1][51:60]]
    
    N_average=round((sum(N_NR)+sum(8-i for i in N_R))/12,2)
    E_average=round((sum(E_NR)+sum(8-i for i in E_R))/12,2)
    O_average=round(sum(O_NR)/12,2)
    A_average=round((sum(A_NR)+sum(8-i for i in A_R))/12,2)
    C_average=round((sum(C_NR)+sum(8-i for i in C_R))/12,2)

    data[1]=data[1]+[N_average,E_average,O_average,A_average,C_average]
    
    with open(os.path.join(new_path,file),'w',encoding='utf-8') as f:
        csvwriter=csv.writer(f)
        csvwriter.writerows(data)

# This is to add the Big-Five personality scores of users to the dialogue files
dialogue_path="dialog1"
dialogue_files=os.listdir(dialogue_path)

persona_path="persona_reorder1"
persona_files=os.listdir(persona_path)
persona=[]
for file in persona_files:
    persona.append(file.split(".")[0])

for dialog in dialogue_files:
    speaker=[]
    with open(os.path.join(dialogue_path,dialog),'r',encoding='utf-8') as f:
        reader=csv.reader(f)
        rows=list(reader)
        header=rows[0]
        
        for i in ["Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"]:
            header.append(i) 
        rows[0]=header
        
        with open(os.path.join(dialogue_path,dialog),'w',encoding='utf-8') as f:
            writer=csv.writer(f)
            writer.writerows(rows)   
        
        speaker=[row[0] for row in rows[1:]]
        
        for s in speaker[:2]:
            for p in persona:
                if s==p:
                    with open(os.path.join(persona_path,p+".csv"),'r',encoding='utf-8') as f:
                        data=read_csv(f)
                        big5=data.iloc[:,60:]
                        big5_values=big5.values.tolist()[0]
                        rows[speaker.index(s)+1]=rows[speaker.index(s)+1]+big5_values
                             
                        with open(os.path.join(dialogue_path,dialog),'w',encoding='utf-8') as f:
                            writer=csv.writer(f)
                            writer.writerows(rows)
                            break                    
                        