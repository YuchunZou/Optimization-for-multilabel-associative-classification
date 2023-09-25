import os,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import pickle
from flask import Flask, jsonify, render_template, request,redirect, url_for
import json
import base64
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

fl = pickle.load(open('./pkl/fl.pkl', 'rb'))
decisionruleset = pickle.load(open('./pkl/drl.pkl', 'rb'))
sg1=pd.read_csv("./data/training.csv")
ad=sg1.drop(columns=['case'])
mtl=pd.read_csv("./data/multiclass.csv")


ac = np.ones(shape=(sg1.shape[0], decisionruleset.shape[0]))
ac = pd.DataFrame(ac, columns=decisionruleset['antecedents'], index=sg1.index).astype(int)
ac['Outcome'] = mtl['Outcome']

for t in range(0, ac.shape[0]):
    for r in range(0, (ac.shape[1] - 1)):
        if any(ad.iloc[t, :] - fl.iloc[r, :] < 0):
            ac.iloc[t, r] = 0

ac1=ac[(ac['Outcome']==1)].drop(columns=['Outcome'])
ac2=ac[(ac['Outcome']==2)].drop(columns=['Outcome'])
ac3=ac[(ac['Outcome']==3)].drop(columns=['Outcome'])
ac4=ac[(ac['Outcome']==4)].drop(columns=['Outcome'])

ac1.loc['Row_sum'] = ac1.apply(lambda x: x.sum())
ac2.loc['Row_sum'] = ac2.apply(lambda x: x.sum())
ac3.loc['Row_sum'] = ac3.apply(lambda x: x.sum())
ac4.loc['Row_sum'] = ac4.apply(lambda x: x.sum())

mtl_df=np.zeros(shape=(decisionruleset.shape[0],8))
mtl_df = pd.DataFrame(mtl_df, index=decisionruleset['antecedents'],columns=['1','2','3','4',
                                                                           'Max_Pro','Prob','Type','ranked_value']).astype(int)

for i in range(0,decisionruleset.shape[0]):
    mtl_df.iloc[i,0]=ac1.loc['Row_sum'][i]*decisionruleset['lift'].iloc[i]/(ac1.shape[0]-1)
    mtl_df.iloc[i,1]=ac2.loc['Row_sum'][i]*decisionruleset['lift'].iloc[i]/(ac2.shape[0]-1)
    mtl_df.iloc[i,2]=ac3.loc['Row_sum'][i]*decisionruleset['lift'].iloc[i]/(ac3.shape[0]-1)
    mtl_df.iloc[i,3]=ac4.loc['Row_sum'][i]*decisionruleset['lift'].iloc[i]/(ac4.shape[0]-1)

for i in range(0,mtl_df.shape[0]):
    mtl_df['Max_Pro'].iloc[i]=mtl_df.iloc[i,0:4].max()
    mtl_df['Prob'].iloc[i]=mtl_df['Max_Pro'].iloc[i]/mtl_df.iloc[i,0:4].sum()


arr=mtl_df.drop(columns=['Max_Pro','Prob','Type'])
for i in range(0,arr.shape[0]):
    mtl_df['Type'].iloc[i] = np.random.choice(arr.iloc[i,:][arr.iloc[i,:] == arr.iloc[i,:].max()].index)

for i in range(0,mtl_df.shape[0]):
    mtl_df['ranked_value'].iloc[i]=decisionruleset['ranked_value'].iloc[i]

mtl_df[['Type']] = mtl_df[['Type']].astype(int)
print(mtl_df)
pickle.dump(mtl_df, open('./pkl/mtl.pkl', 'wb'))
#print(fl)
#print(drl)
#drl.to_csv('./data/decisionruleset.csv', index = False)
#fl.to_csv('./data/featurelist.csv', index = False)