from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from __future__ import division
import string
import math
import csv
import random
import time
import pandas as pd


def get_std_slotprice(path,column="slotprice"):
    df = pd.read_csv(path, skipinitialspace=True, usecols=[column])
    return int(df.slotprice.values.std())

def get_LRS_params(path):
    df=pd.read_csv(path)
    avgCTR=(df.click.sum()/df.shape[0])*100
    base_bid=df.payprice.mean()
    return avgCTR,base_bid

def load_data(filepath,training=True):
    data = defaultdict(list)
    labels = defaultdict(list)
    # std of slotprice for nomalization
    STD_SLOTPRICE = get_std_slotprice(filepath,column="slotprice")
    print "std stored"
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # pass header:
        next(reader)
        # Iterate:
        for row in reader:
            instance=process_event(row,STD_SLOTPRICE,training)
            data[row[24]].append(instance)
            labels[row[24]].append(int(row[0]))
    print "data and labels loaded"
    return data,labels


def trainCTRmodel(training_events, training_labels):
    models = {}
    for key in training_events.keys():
        data=training_events[key]
        labels= training_labels[key]


        label_encoder = LabelEncoder()
        vectorizer = DictVectorizer()

        train_event_x = vectorizer.fit_transform(data)
        train_event_y = label_encoder.fit_transform(labels)

        # Getting the class weight to rebalance data:
        neg_weight = sum(labels) / len(labels)
        pos_weight = 1 - neg_weight

        # Create and train the model.
        p = 0.34
        lr = LogisticRegression(C=p, class_weight={1: pos_weight, 0: neg_weight})
        #print "labels: "+str(labels)
        lr.fit(train_event_x, train_event_y)
        models[key] = (lr, label_encoder, vectorizer)
        print('Training model for advertiser %s done')%(key)
    return models


def process_event(row,STD_SLOTPRICE,training=True):
    # Initilize instance:
    if training==True:
        instance = {'weekday': row[1], 'hour': row[2], 'region': row[8], \
                    'city': row[9], 'adexchange': row[10], 'slotwidth': row[15], 'slotheight': row[16], \
                    'slotvisibility': row[17], 'slotformat': row[18], 'slotprice': float(row[19]) / STD_SLOTPRICE, \
                    'advertiser': row[24]}
    else:
        instance = {'weekday': row[1], 'hour': row[2], 'region': row[8], \
                    'city': row[9], 'adexchange': row[10], 'slotwidth': row[15], 'slotheight': row[16], \
                    'slotvisibility': row[17],'payprice':int(row[22]), 'slotformat': row[18], 'slotprice': float(row[19]) / STD_SLOTPRICE, \
                    'advertiser': row[24]}

    # Add usertags:
    usertags = row[25].split(',')
    temp_dict = {}
    for tag in usertags:
        temp_dict["tag " + tag] = True
    instance.update(temp_dict)
    # add OS and browser:
    op_sys, browser = row[6].split('_')
    instance.update({op_sys: True, browser: True})
    return instance

def predict_event_CTR(instance,advertiser, model): # models:dict
    #print "model adv " +str(model[advertiser])
    lr = model[advertiser][0]
    # Transform event:
    label_encoder = model[advertiser][1]
    vectorizer = model[advertiser][2]
    event = [instance]
    event_x = vectorizer.transform(event)
    #event_y = label_encoder.inverse_transform(lr.predict(event_x))
    event_y = lr.predict_proba(event_x)
    return event_y[0][1]


def RTB_simulation(model, validation_path, training_path,
                   start_budget = 25000000, lambda_const=5.2e-7, c=20):  # param is the dictionary with the bidprice per advertiser
    impressions = 0
    clicks = 0
    #number_rows=0
    budget=start_budget
    # Calculate the standard deviation for slotprice
    STD_SLOTPRICE = get_std_slotprice(validation_path)

    # Linear Stragegy:
    avgCTR,base_bid = get_LRS_params(training_path)


    with open(validation_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            advertiser=row[24]
            # parsing event:
            instance = process_event(row, STD_SLOTPRICE, training=False)

            # Predicting CTR, in this case this is our b(theta):
            pCTR = predict_event_CTR(instance,advertiser, model)
            #print "pCTR: "+str(pCTR)
            # Calculate the bid based on ORTB:
            #current_bid = np.sqrt((c*pCTR)/lambda_const+np.power(c,2))-c

            # Second way:
            current_bid=np.power((pCTR+np.square((c**2*lambda_const**2+pCTR**2))/(c*lambda_const)),(1/3))- \
                                np.power(((c*lambda_const)/(pCTR+np.square((c**2*lambda_const**2+pCTR**2)))),(1/3))

            # Check if we still have budget:
            if budget > current_bid:

                # Get the market price:
                payprice = instance['payprice']

                # Check if we win the bid:
                if current_bid > payprice:
                    impressions += 1
                    budget -= payprice
                    # Check if the person clicks:
                    if row[0] == "1":
                        print "current bid : %d , payprice: %d, click? : %s" % (int(current_bid), int(payprice), row[0])
                        clicks += 1


    print("Impressions:{0}".format(impressions))
    print("Clicks:{0}".format(clicks))
    print("Reamining Budget:{0}".format(budget))
    if impressions > 0:
        print "Best bid CTR: " + str((clicks / impressions) * 100)
        return (clicks / impressions) * 100
    else:
        return -1





if __name__=="__main__":
    # MAIN:
    st=time.time()
    training_path = r"../dataset/train.csv"
    validation_path = r"../dataset/validation.csv"

    # Extracting data:
    training_events_best, labels_best = load_data(training_path)

    # training model
    models_best_CTR= trainCTRmodel(training_events_best, labels_best)

    val_best_CTR=RTB_simulation(models_best_CTR, validation_path, training_path,c=80)

    print time.time()-st
