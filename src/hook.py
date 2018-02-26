# -*- coding: utf-8 -*-
from evaluate import strict, loose_macro, loose_micro
import numpy as np
def remove_conflict(tags,scores,hier):
    tags_new = [] 
    tags = sorted([t for t in tags],key=lambda x:scores[x])[::-1]

    for tag in tags:
        path_t = set(hier[tag].nonzero()[0])
        keep = True
        for tag2 in tags_new:
            path_t2 = set(hier[tag2].nonzero()[0])	
            if not (path_t2<path_t or path_t<path_t2):
                keep=False
        if keep:
            tags_new.append(tag)
    return tags_new
def acc_hook(scores, y_data,other_id):
    true_and_prediction = []
    corr = 0.
    total = 0. 
    for score,true_label in zip(scores,y_data):
        predicted= []
        true_tag = []
        for i in range(len(score)):
            t1 = true_label[i]
            if t1 != 0 :
                true_tag.append(i)
            t2 = np.argmax(score[i])
            if t2 != 0:
                predicted.append(i)
            if t1 != 0: 
                total += 1
                t2 = np.argmax(score[i][1:]) + 1
                if t1 == t2: 
                    corr +=1.0
        if predicted ==[]:predicted = [other_id]
        if true_tag == []: true_tag = [other_id]
        true_and_prediction.append((true_tag, predicted))
    stct= strict(true_and_prediction)
    macro = loose_macro(true_and_prediction)
    micro = loose_micro(true_and_prediction)
    report = []
    if total != 0:
        report.append("Accuracy:%f %d" % (corr/total,total))
    report.append("     strict (p,r,f1):%f %f %f:" % stct)
    report.append("loose macro (p,r,f1):%f %f %f" % macro)
    report.append("loose micro (p,r,f1):%f %f %f" % micro)
    report.append("%f\t%f\t%f" % (stct[-1],macro[-1],micro[-1]))
    return stct[2],report,[t[1] for t in true_and_prediction]

