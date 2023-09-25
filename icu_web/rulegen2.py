import pandas as pd
import numpy as np
from gurobipy import *
import pickle

from numpy import linalg as la
import string
from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import statistics
from sklearn import tree
import warnings
warnings.filterwarnings("ignore")
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori

sg1=pd.read_csv("./data/training.csv")
frequent_itemsets_positive=apriori(sg1, min_support = 0.01,use_colnames = True, max_len=5)
rules_positive=association_rules(frequent_itemsets_positive,metric='confidence',min_threshold=0.7)

rules_positive=rules_positive[(rules_positive['consequents']==frozenset(['case']))]
ruleset_list=rules_positive.reset_index(drop=True)

st_rulesets=pd.DataFrame(ruleset_list)
st_ruleset=st_rulesets.drop(columns=['antecedent support','consequent support','leverage','conviction'])

rset = np.zeros(shape=(st_ruleset.shape[0],sg1.shape[1]))
rset = pd.DataFrame(rset, columns = sg1.columns).astype(int)

s=st_ruleset.copy()
s=s.reset_index(drop=True)

for i in list(sg1.columns):
    rset.loc[s['antecedents'].astype(str).str.contains(i) | s['consequents'].astype(str).str.contains(i),i] =1

rsetT=rset.T
rsetT=rsetT.drop(['case'])
rsetT.columns=st_ruleset['antecedents']

Dys= np.ones(shape=(st_ruleset.shape[0],st_ruleset.shape[0]))
Dys= pd.DataFrame(Dys, index= st_ruleset['antecedents'],columns = st_ruleset['antecedents']).astype(int)

for i in range(rsetT.shape[1]):
    for j in range(rsetT.shape[1]):
        if any(rsetT.iloc[:,i]-rsetT.iloc[:,j]<0):
              Dys.iloc[i][j]=0

Dys['redudant_del']  = 0
for i in range(Dys.shape[0]):
    if(sum(Dys.iloc[i,:])> 1):
        Dys.iloc[i,Dys.shape[0]]=1

Dys_prunned=Dys[(Dys['redudant_del']==0)]
select_list=[]

for i in range(Dys_prunned.shape[0]):
    for j in range(st_ruleset.shape[0]):
        if(list(Dys_prunned.index)[i] == list(st_ruleset['antecedents'])[j]):
            select_list.append(j)

selected_list = np.array(select_list)
selected_ruleset=st_rulesets.iloc[selected_list,:]

sorted_ruleset=selected_ruleset.drop(columns=['antecedent support','consequent support','leverage','conviction'])

# ranking algorithm
sorted_ruleset['lenT'] = 0

for i in range(sorted_ruleset.shape[0]):
    sorted_ruleset['lenT'].iloc[i] = 1 / (sorted_ruleset['antecedents'].astype(str).iloc[i].count(',') + 1)

sorted_ruleset['ranked_value']=0

sorted_ruleset=sorted_ruleset.sort_values(by=['lift','support','lenT'], ascending=False)

sorted_ruleset['ranked_value']=0

sorted_ruleset=sorted_ruleset.sort_values(by=['lift','support','lenT'], ascending=False)

for i in range(sorted_ruleset.shape[0]):
    #sorted_ruleset['ranked_value'].iloc[i]=1+(sorted_ruleset.shape[0]-i)/sorted_ruleset.shape[0]
    sorted_ruleset['ranked_value'].iloc[i]=1+(i+1)/sorted_ruleset.shape[0]

sorted_rulesetT=sorted_ruleset
sorted_ruleset=sorted_ruleset.drop(columns=['lenT'])

for i in range(sorted_rulesetT.shape[0]):
     sorted_rulesetT['lenT'].iloc[i] = (1/sorted_rulesetT['lenT'].iloc[i]).astype(int)
rk=sorted_ruleset['ranked_value'].values

#Generate binary representation for rules--- b
Dummy_positive = np.zeros(shape=(sorted_ruleset.shape[0],sg1.shape[1]))
B = pd.DataFrame(Dummy_positive, columns = sg1.columns).astype(int)

s=sorted_ruleset.copy()
s=s.reset_index(drop=True)

for i in list(sg1.columns):
    B.loc[s['antecedents'].astype(str).str.contains(i) | s['consequents'].astype(str).str.contains(i),i] =1

b=B.T
b=b.drop(['case'])
b.columns=sorted_ruleset['antecedents']

Dummy_positive_c = np.ones(shape=(sg1.shape[0],sorted_ruleset.shape[0]))
Dummy_c = pd.DataFrame(Dummy_positive_c, columns =sorted_ruleset['antecedents'],index=sg1.index).astype(int)

a=sg1
a1=a.drop(columns=['case'])
B1=B.drop(columns=['case'])

for t in range(0,Dummy_c.shape[0]):
    for r in range(0,Dummy_c.shape[1]):
        if any(a1.iloc[t,:]-B1.iloc[r,:]<0):
              Dummy_c.iloc[t][r]=0

case=pd.DataFrame(a['case'],columns=['case'])
c = pd.concat([case, Dummy_c], axis=1)

cr=c.copy()

for i in range(0,c.shape[0]):
    for j in range(1,c.shape[1]):
        if(c.iloc[i,j]==1):
            cr.iloc[i,j]=sorted_ruleset['ranked_value'].iloc[(j-1)]
#ARSOM
c_P = cr.loc[cr['case']==1]
c_N = cr.loc[cr['case']==0]

c_P=c_P.drop(columns=['case'])
c_N=c_N.drop(columns=['case'])

##Num of Column
##Num of Row
K=c_N.shape[1]
J=b.shape[0]
I=cr.shape[0]
I_P=c_P.shape[0]
I_N=c_N.shape[0]

af=a.values
bf=b.values
cf=cr.values
c_P_df=c_P.values
c_N_df=c_N.values

#Groubi
A = Model('assing_problem')
y = {}
for j in range(J):
    y[j] = A.addVar(lb=0,ub=1,vtype=GRB.BINARY, name="y")

z = {}
for k in range(K):
    z[k] = A.addVar(lb=0,ub=1,vtype=GRB.BINARY, name="z")

x = {}
for i in range(I):
    x[i] = A.addVar(lb=0,ub=1,vtype=GRB.BINARY, name="x")

A.update()
A.setObjective(quicksum(y[j] for j in range(J))+quicksum(z[k] for k in range(K))+quicksum(x[i] for i in range(I_P,I))-999*quicksum(x[i] for i in range(I_P)),GRB.MINIMIZE)
for i in range(I_P):
    A.addConstr(quicksum(c_P_df[i][k] * z[k] for k in range(K))>= x[i], name="con1")

for i in range(I_P,I):
    i1=i-I_P
    A.addConstr(quicksum(c_N_df[i1][k] * z[k] for k in range(K))<= (K+1) * x[i], name="con2")

for j in range(J):
    A.addConstr(quicksum(bf[j][k] * z[k] for k in range(K))<= (K+1) * y[j], name="con3")

A.optimize()

n=0
Zlist=[]
if   A.Status == GRB.OPTIMAL:
    for var in A.getVars():
        if var.varName=="z":
            Zlist.append(var.X)
            print('%s %g'%(var.varName, var.X))
            if var.X==1:
                n+=1

Z=pd.DataFrame(list(map(int,Zlist)),columns=['z'])
sorted_ruleset1=sorted_ruleset.reset_index(drop=True)
decisionrule = pd.concat([sorted_ruleset1, Z], axis=1)
decisionrule_list=decisionrule[(decisionrule['z']==1)]
decisionruleset=decisionrule_list.drop(columns=['z'])
with open('./pkl/drl.pkl', 'wb') as file:
    pickle.dump(decisionruleset, file)
print(decisionruleset)
