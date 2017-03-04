from __future__ import division
import pandas as pd
import numpy as np
import time

def win_loss_count(bidding_price,payprice):
    #print "bidding_price: "+str(bidding_price)
    #print "payprice: " + str(payprice)
    if bidding_price>payprice:
        return 1
    else:
        return 0



def count_wins(advertiser,adver_win_count,bidding_price,payprice):
    #print "adver: "+str(advertiser) + " "+str(adver_win_count)
    if advertiser not in adver_win_count.keys():
        adver_win_count[advertiser]=win_loss_count(bidding_price,payprice)
    else:
        adver_win_count[advertiser]+=win_loss_count(bidding_price,payprice)



def evaluate_constant_bid(validation_set,bidding_strategie):

    # Given a validation set and constant bidprice strategy
    k=0
    winning_bids={}

    # Iterating over all advertisers and their payprice:
    for advertiser,payprice in validation_set[["advertiser", "payprice"]].values:
        k+=1
        bidding_price=bidding_strategie[advertiser]
        #print "bidding_price: "+str(bidding_price)
        #print "payprice: " + str(payprice)
        #print "winn: "+str(winning_bids)
        count_wins(advertiser,winning_bids,bidding_price,payprice)
        #print "winning_bids: "+str(winning_bids)
        if k%10000==0:
            print "%d completed" %(k)
    return winning_bids


if __name__=="__main__":
    time_start = time.clock()

    # Importing data:
    filepath="../dataset/train.csv"
    training_data=pd.read_csv(filepath)

    # Taking positive clicks
    clicks_training=training_data[training_data["click"] == 1]

    # Constant bid set to the average of the payprice
    constant_bid = dict(clicks_training.groupby("advertiser").mean()["payprice"])

    # Importing validation path:
    validation_path="../dataset/validation.csv"
    validation_set=pd.read_csv(validation_path)

    # testing constant bid strategie:


    winning_bids = evaluate_constant_bid(validation_set, constant_bid)
    total_bids = dict(validation_set.groupby("advertiser").sum()["logtype"])



    print clicks_training.shape



    # time:
    time_elapsed = (time.clock() - time_start)
    print time_elapsed

