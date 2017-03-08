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

# Constant strategy
def get_constant_bid(filepath):
    payprice_click_df = pd.read_csv(filepath, usecols=["payprice","click"])
    #print "training data imported: %d rows, %d columns"%(training_data.shape[0], training_data.shape[1])

    # Taking payprice mean of positive clicks:
    avg_winning_pp = payprice_click_df[payprice_click_df["payprice"]>0].mean()["payprice"]
    max_winning_pp = payprice_click_df[payprice_click_df["payprice"] > 0].max()["payprice"]

    b= abs(max_winning_pp-avg_winning_pp)/4
    constant_bid = avg_winning_pp + b
    print "constant bid: %0.2f" %constant_bid
    return int(constant_bid)

def get_random_bid(filepath):
    payprice_click_df = pd.read_csv(filepath, usecols=["payprice", "click"])
    avg_winning_pp = payprice_click_df[payprice_click_df["payprice"] > 0].mean()["payprice"]
    max_winning_pp = payprice_click_df[payprice_click_df["payprice"] > 0].max()["payprice"]
    #std_winning_pp = payprice_click_df[payprice_click_df["payprice"] > 0].std()["payprice"]
    b = abs(max_winning_pp - avg_winning_pp) / 4

    min_pp = avg_winning_pp +b#+ std_winning_pp
    max_pp = avg_winning_pp +b+10 #2*std_winning_pp

    return int(min_pp),int(max_pp)


def process_event(row):
    instance = {'bidprice': int(row[21]), 'payprice': int(row[22])}
    return instance

def RTB_simulation(mode, validation_path, training_path, start_budget = 25000):  # param is the dictionary with the bidprice per advertiser
    impressions = 0
    clicks = 0
    budget=start_budget

    # Stragegies:
    if mode == 'constant':
        # Calculating constant bid:
        constant_bid=get_constant_bid(training_path)
        # Iterating over each new impression:
        with open(validation_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)

            for row in reader:
                #print "row"
                #conts_payprice = 4000  # param[advertiser]
                if budget > constant_bid:
                    instance = process_event(row)
                    payprice = instance['payprice']
                    if constant_bid > payprice:
                        impressions += 1
                        budget -= payprice
                        #print "budget %d" %budget
                        if row[0] == "1":
                            clicks += 1

    if mode == 'random':
        min_bid,max_bid= get_random_bid(training_path)
        #print "span: %d - %d "%(min_bid,max_bid)

        with open(validation_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)

            for row in reader:
                current_bid = random.randrange(min_bid,max_bid)
                #print "current_bid: %d budget: %d"%(current_bid,budget)
                if budget > current_bid:
                    #payprice = int(row[22])
                    instance = process_event(row)
                    payprice = instance['payprice']
                    #print "current_bid - payprice : %d"%(current_bid - payprice)
                    if current_bid > payprice:
                        impressions += 1
                        budget -=payprice
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
    #training_events, labels = load_data(training_path)
    print time.time()-st

    CTR=RTB_simulation('random', validation_path, training_path, 25000)

    print "time: " +str(time.time()-st)