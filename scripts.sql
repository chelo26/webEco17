Stuff its doing right now:

#Impressions, clicks and CTR per...
#Weekday
SELECT	t.weekday, 
		count(*) AS impressions,
		SUM(t.click = '1') AS clicks, 
		SUM(t.click = '1')/ CAST(count(*) as real)*100 AS CTR
FROM train t
GROUP BY t.weekday

#Maximum and minimum payprice
SELECT weekday, max(payprice)
from train

#Maximum and minimum bidprice in total

#Get frequencies of how much they paid. Put the things in buckets. This for advertisers





