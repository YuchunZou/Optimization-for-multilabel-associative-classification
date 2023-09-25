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

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index copy.html")

@app.route('/predict',methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        img_stream = ''
        img_path = './image/auc.png'
        with open(img_path, 'rb') as img_f:
            img_stream = base64.b64encode(img_f.read()).decode("ascii")

        mydata = pd.DataFrame(
            {'TRIAGE1': [request.form["TRIAGE1"]], 'TRIAGE2': [1-int(request.form["TRIAGE1"])],
             'elderly': [request.form["elderly"]], 'DM': [request.form["DM"]],
             'HTN': [request.form["HTN"]], 'CAD': [request.form["CAD"]], 'CVA': [request.form["CVA"]],
             'cls_CPC': [request.form["cls_CPC"]],
             'RESP': [request.form["RESP"]], 'CHF': [request.form["CHF"]], 'LC': [request.form["LC"]],
             'ESRD': [request.form["ESRD"]], 'CANCE': [request.form["CANCE"]],
             'IMMUN': [request.form["IMMUN"]],
             'ER_USV21': [request.form["ER_USV21"]], 'ER_USV25': [request.form["ER_USV25"]],
             'ER_USV24': [request.form["ER_USV24"]], 'ER_USV20': [request.form["ER_USV20"]],
             'ER_USVlowBP': [request.form["ER_USVlowBP"]], 'ER_desat': [request.form["ER_desat"]],
             'ER_USV14': [request.form["ER_USV14"]],
             'ER_USV15': [request.form["ER_USV15"]], 'ER_USVhema': [request.form["ER_USVhema"]],
             'ER_USVmeta': [request.form["ER_USVmeta"]],
             'ER_USV_RSD': [request.form["ER_USV_RSD"]], 'ER_USV7': [request.form["ER_USV7"]],
             'ER_USV8': [request.form["ER_USV8"]], 'ER_USV9': [request.form["ER_USV9"]],
             'ER_USV10': [request.form["ER_USV10"]], 'ER_USV11': [request.form["ER_USV11"]]
             }).astype(int)
        fl = pickle.load(open('./pkl/fl.pkl', 'rb'))
        drl = pickle.load(open('./pkl/drl.pkl', 'rb'))

        #Overall Patient group
        Dummy_check = pd.DataFrame(np.ones(shape=(mydata.shape[0], drl.shape[0]))).astype(int)

        for r in range(0, (Dummy_check.shape[1])):
            if any(mydata.iloc[0, :] - fl.iloc[r, :] < 0):
                Dummy_check.iloc[0, r] = 0

        ac=Dummy_check
        a = Dummy_check.iloc[0,].sum()

        for j in range(0, Dummy_check.shape[1]):
            if (Dummy_check.iloc[0, j] == 1):
                Dummy_check.iloc[0, j] = drl['confidence'].iloc[j]

        pos_prob = round((Dummy_check.iloc[0,].sum() / a),3)
        neg_prob=round((1-pos_prob),3)

        # Multiclass Patient Group
        mtl = pickle.load(open('./pkl/mtl.pkl', 'rb'))
        ac_num = ac.copy()
        ac_prob = ac.copy()

        for j in range(0, (ac_num.shape[1])):
            ac_num.iloc[0, j] = mtl['Type'].iloc[j]
            ac_prob.iloc[0, j] = mtl['Prob'].iloc[j] * mtl['ranked_value'].iloc[j]

        checklist = np.zeros(shape=(ac.shape[0], 12))
        checklist = pd.DataFrame(checklist,
                                 columns=['1', '2', '3', '4', 'P1', 'P2', 'P3', 'P4', 'NP1', 'NP2', 'NP3',
                                          'NP4']).astype(int)

        checklist.iloc[0, 0] = list(ac_num.iloc[0, :]).count(1)
        checklist.iloc[0, 1] = list(ac_num.iloc[0, :]).count(2)
        checklist.iloc[0, 2] = list(ac_num.iloc[0, :]).count(3)
        checklist.iloc[0, 3] = list(ac_num.iloc[0, :]).count(4)

        for j in range(0, ac_num.shape[1]):
            if (ac_num.iloc[0, j] == 1):
                checklist.iloc[0, 4] += ac_prob.iloc[0, j]
            elif (ac_num.iloc[0, j] == 2):
                checklist.iloc[0, 5] += ac_prob.iloc[0, j]
            elif (ac_num.iloc[0, j] == 3):
                checklist.iloc[0, 6] += ac_prob.iloc[0, j]
            elif (ac_num.iloc[0, j] == 4):
                checklist.iloc[0, 7] += ac_prob.iloc[0, j]

        if (checklist.iloc[0, 4:8].sum() != 0):
            checklist.iloc[0, 8] = checklist.iloc[0, 4] / checklist.iloc[0, 4:8].sum()
            checklist.iloc[0, 9] = checklist.iloc[0, 5] / checklist.iloc[0, 4:8].sum()
            checklist.iloc[0, 10] = checklist.iloc[0, 6] / checklist.iloc[0, 4:8].sum()
            checklist.iloc[0, 11] = checklist.iloc[0, 7] / checklist.iloc[0, 4:8].sum()

        # Subgroup1
        g1_pos = round(checklist.iloc[0, 8], 3)
        g1_neg = round((1 - g1_pos), 3)
        # Subgroup2
        g2_pos = round(checklist.iloc[0, 9], 3)
        g2_neg = round((1 - g2_pos), 3)
        # Subgroup3
        g3_pos = round(checklist.iloc[0, 10], 3)
        g3_neg = round((1 - g3_pos), 3)
        # Subgroup4
        g4_pos = round(checklist.iloc[0, 11], 3)
        g4_neg = round((1 - g4_pos), 3)

    return render_template("index copy.html", results=pos_prob, neg_results=neg_prob, g1_results=g1_pos,
                               g1_neg_results=g1_neg,
                               g2_results=g2_pos, g2_neg_results=g2_neg, g3_results=g3_pos, g3_neg_results=g3_neg,
                               g4_results=g4_pos, g4_neg_results=g4_neg, img_stream=img_stream)

if __name__ == '__main__':
    app.run(debug=True)