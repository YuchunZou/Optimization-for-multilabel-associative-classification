import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_auc_score,roc_curve, auc


mtl = pickle.load(open('./pkl/mtl.pkl', 'rb'))
mtl_df=pd.read_csv("./data/multiclass.csv")
sg1=pd.read_csv("./data/training.csv")
wholedata=sg1
wholedata['Outcome']=mtl_df['Outcome']
train,test = train_test_split(wholedata,test_size=0.2)
test=test.reset_index(drop=True)
ad = test.drop(columns=['case','Outcome'])
fl = pickle.load(open('./pkl/fl.pkl', 'rb'))
decisionruleset = pickle.load(open('./pkl/drl.pkl', 'rb'))

Dummy_check = np.ones(shape=(test.shape[0],decisionruleset.shape[0]))
Dummy_check = pd.DataFrame(Dummy_check, columns =decisionruleset['antecedents'],index=test.index).astype(int)
Dummy_check['probaility']=0
Dummy_check['case']=test['case']

for t in range(0,Dummy_check.shape[0]):
    for r in range(0,(Dummy_check.shape[1]-2)):
        if any(ad.iloc[t,:]-fl.iloc[r,:]<0):
              Dummy_check.iloc[t,r]=0


ac=Dummy_check.copy().drop(columns=['probaility','case'])
Dummy_check1=Dummy_check.copy()
Dummy_check1=Dummy_check1.drop(columns=['probaility','case'])


for i in range(0,Dummy_check1.shape[0]):
    for j in range(0,Dummy_check1.shape[1]):
        if(Dummy_check1.iloc[i,j]==1):
            a=sum(Dummy_check1.iloc[i,:])
            Dummy_check.iloc[i,j]=decisionruleset['confidence'].iloc[j]
            Dummy_check1.iloc[i,j]=decisionruleset['confidence'].iloc[j]
            Dummy_check['probaility'].iloc[i]=sum(Dummy_check1.iloc[i,:])/a

print(roc_auc_score(Dummy_check['case'].values, Dummy_check['probaility'].values))



ac_num=ac.copy()
ac_prob=ac.copy()

for i in range(0,ac_num.shape[0]):
    for j in range(0,(ac_num.shape[1])):
        if(ac_num.iloc[i,j]==1):
            ac_num.iloc[i,j]=mtl['Type'].iloc[j]
            ac_prob.iloc[i,j]=mtl['Prob'].iloc[j]*mtl['ranked_value'].iloc[j]

checklist=np.zeros(shape=(ac.shape[0],16))
checklist = pd.DataFrame(checklist,columns=['1','2','3','4','P1','P2','P3','P4','NP1','NP2','NP3','NP4','C1','C2','C3','C4']).astype(int)

for i in range(0,ac_num.shape[0]):
        checklist.iloc[i,0]=list(ac_num.iloc[i,:]).count(1)
        checklist.iloc[i,1]=list(ac_num.iloc[i,:]).count(2)
        checklist.iloc[i,2]=list(ac_num.iloc[i,:]).count(3)
        checklist.iloc[i,3]=list(ac_num.iloc[i,:]).count(4)

for i in range(0,ac_num.shape[0]):
    for j in range(0,ac_num.shape[1]):
        if (ac_num.iloc[i,j]==1):
            checklist.iloc[i,4]+=ac_prob.iloc[i,j]
        elif (ac_num.iloc[i,j]==2):
            checklist.iloc[i,5]+=ac_prob.iloc[i,j]
        elif (ac_num.iloc[i,j]==3):
            checklist.iloc[i,6]+=ac_prob.iloc[i,j]
        elif (ac_num.iloc[i,j]==4):
            checklist.iloc[i,7]+=ac_prob.iloc[i,j]

for i in range(0,checklist.shape[0]):
    if(checklist.iloc[i,4:8].sum()!=0):
        checklist.iloc[i,8]=checklist.iloc[i,4]/checklist.iloc[i,4:8].sum()
        checklist.iloc[i,9]=checklist.iloc[i,5]/checklist.iloc[i,4:8].sum()
        checklist.iloc[i,10]=checklist.iloc[i,6]/checklist.iloc[i,4:8].sum()
        checklist.iloc[i,11]=checklist.iloc[i,7]/checklist.iloc[i,4:8].sum()

checklist['Outcome']=test['Outcome']
for i in range(0, checklist.shape[0]):
    if (checklist['Outcome'].iloc[i] == 1):
        checklist['C1'].iloc[i] = 1
    if (checklist['Outcome'].iloc[i] == 2):
        checklist['C2'].iloc[i] = 1
    if (checklist['Outcome'].iloc[i] == 3):
        checklist['C3'].iloc[i] = 1
    if (checklist['Outcome'].iloc[i] == 4):
        checklist['C4'].iloc[i] = 1

checklist = checklist.drop(columns=['Outcome'])
print(roc_auc_score(checklist['C1'].values, checklist['NP1'].values))
print(roc_auc_score(checklist['C2'].values, checklist['NP2'].values))
print(roc_auc_score(checklist['C3'].values, checklist['NP3'].values))
print(roc_auc_score(checklist['C4'].values, checklist['NP4'].values))




# Calculate every type auc
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(Dummy_check['case'].values, Dummy_check['probaility'].values)
fpr[1], tpr[1], _ = roc_curve(checklist['C1'].values, checklist['NP1'].values)
fpr[2], tpr[2], _ = roc_curve(checklist['C2'].values, checklist['NP2'].values)
fpr[3], tpr[3], _ = roc_curve(checklist['C3'].values, checklist['NP3'].values)
fpr[4], tpr[4], _ = roc_curve(checklist['C4'].values, checklist['NP4'].values)
for i in range(5):
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(5):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= 5

# Plot all ROC curves
lw = 2
plt.figure()
colors = cycle(['red','green', 'darkorange', 'cornflowerblue', 'purple'])
for i, color in zip(range(5), colors):
    if (i==0):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of overall class (area = {revisiontag})'.format(revisiontag=round(roc_auc[i],4)))
    else:
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('the icu datasets for overall and multi-class classification')
plt.legend(loc="lower right")
plt.show()
plt.savefig('./image/auc.png')
