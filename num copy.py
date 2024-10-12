import pandas as pd
import numpy as np

alist = []
for i in range(0, 5):
    a = str(i)
    b = "logs/00/46b4470/att_combination5_option4_last_6/20230112T162116/mimic3.fold_"+a+"_all.csv"
    # print(b)
    data = pd.read_csv(b)
    df = data.sort_values(by='test_f1', ascending=False)
    vals = df[0:2]
    print(vals.to_numpy())
    alist.append(vals)
    # vals.to_csv('test.csv')

print(alist)
results = np.concatenate(alist, 0) 
print(results)
r=''
for idx in range(results.shape[1]):
    # print(results)
    # print(results.shape[1])
    # print(np.mean(results[:, idx]))
    print(",",round(np.mean(results[:, idx]),3),"±",round(np.std(results[:, idx]),3))
    r=r+str(round(np.mean(results[:, idx]),3))+"±"+str(round(np.std(results[:, idx]),3))+','
print(r)
    # print(np.std(results[:, idx]))


# for j in range(1,6):
#     a = str(i)
#     c1 = "test"+a+".csv"
#     d = pd.read_csv(c1,nrows=2)  
#     print(d)
    
    