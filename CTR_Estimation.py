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

def load_data(filepath):
    data = list()
    labels = list()
    # std of slotprice for nomalization
    STD_SLOTPRICE = get_std_slotprice(filepath,column="slotprice")
    print "std stored"
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # pass header:
        next(reader)
        # Iterate:
        for row in reader:
            instance=process_event(row,STD_SLOTPRICE)
            data.append(instance)
            labels.append(int(row[0]))
    print "data and labels loaded"
    return data,labels


def train(training_data, labels):
    bidprices = {}
    models = {}

    label_encoder = LabelEncoder()
    vectorizer = DictVectorizer()

    train_event_x = vectorizer.fit_transform(training_data)
    train_event_y = label_encoder.fit_transform(labels)

    # Getting the class weight to rebalance data:
    neg_weight = sum(labels) / len(labels)
    pos_weight = 1 - neg_weight

    # Create and train the model.
    p = 0.34
    lr = LogisticRegression(C=p, class_weight={1: pos_weight, 0: neg_weight})
    lr.fit(train_event_x, train_event_y)
    model = (lr, label_encoder, vectorizer)
    print('Training done')
    return model, train_event_x


def process_event(row,STD_SLOTPRICE):
    # Initilize instance:
    instance = {'weekday': row[1], 'hour': row[2], 'region': row[8], \
                'city': row[9], 'adexchange': row[10], 'slotwidth': row[15], 'slotheight': row[16], \
                'slotvisibility': row[17], 'slotformat': row[18], 'slotprice': float(row[19]) / STD_SLOTPRICE, \
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

def predict_event_labels(instance, model): # models:dict
    lr = model[0]
    # Transform event:
    label_encoder = model[1]
    vectorizer = model[2]
    event = [instance]
    event_x = vectorizer.transform(event)
    #event_y = label_encoder.inverse_transform(lr.predict(event_x))
    event_y = lr.predict_proba(event_x)
    return event_y


def RTB_simulation_linear(model, validation_path, training_path, start_budget = 25000):  # param is the dictionary with the bidprice per advertiser
    impressions = 0
    clicks = 0
    budget=start_budget
    # Calculate the standard deviation for slotprice
    STD_SLOTPRICE = get_std_slotprice(validation_path)

    # Linear Stragegy:

    #min_bid,max_bid= get_random_bid(training_path)
    min_bid, max_bid = 6,31
    #print "span: %d - %d "%(min_bid,max_bid)

    with open(validation_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:

            # Check if we still have budget:
            if budget > current_bid:
                instance = process_event(row,STD_SLOTPRICE)

                # Calculate the pCTR:
                pCTR=predict_event_labels(instance, model)

                # Calculate the bid:
                current_bid = base_bid*pCTR/avgCTR
                ## Complete avgCTR and base_bid
                ##------------------------------

                # Get the market price:
                payprice = instance['payprice']

                # Check if we win the bid:
                if current_bid > payprice:
                    impressions += 1
                    budget -= payprice
                    # Check if the person clicks:
                    if row[0] == "1":
                        clicks += 1


    print("Impressions:{0}".format(impressions))
    print("Clicks:{0}".format(clicks))
    if impressions > 0:
        return (clicks / impressions) * 100
    else:
        return 0





if __name__=="__main__":
    # MAIN:
    st=time.time()
    training_path = r"../dataset/train.csv"
    validation_path = r"../dataset/validation.csv"

    # Extracting data:
    training_events, labels = load_data(training_path)

    # training model
    model,train_vec_x = train(training_events, labels)



    print time.time()-st
