#Impressions, clicks and CTR per...
#Weekday
SELECT	t.weekday, 
		count(*) AS Impressions,
		SUM(t.click = '1') AS Clicks, 
		SUM(t.click = '1')/ CAST(count(*) as real)*100 AS CTR
FROM train t
GROUP BY t.weekday

#Maximum and minimum payprice and bidprice
SELECT weekday, max(payprice), min(payprice), max(bidprice), min(bidprice)
from train
group by weekday

#Bid price per advertiser
select 
		advertiser,
		sum(bidprice >= 0 AND bidprice < 50) as '0-49', 
		sum(bidprice >= 50 AND bidprice < 100) as '50-99',
		sum(bidprice >= 100 AND bidprice < 150) as '100-149',
		sum(bidprice >= 150 AND bidprice < 199) as '150-199',
		sum(bidprice >= 200 AND bidprice < 249) as '200-249',
		sum(bidprice >= 250 AND bidprice <= 300) as '250-300'
from train
group by advertiser

#Pay price per advertiser
select 
		advertiser,
		sum(payprice >= 0 AND payprice < 50) as '0-49', 
		sum(payprice >= 50 AND payprice < 100) as '50-99',
		sum(payprice >= 100 AND payprice < 150) as '100-149',
		sum(payprice >= 150 AND payprice < 199) as '150-199',
		sum(payprice >= 200 AND payprice < 249) as '200-249',
		sum(payprice >= 250 AND payprice <= 300) as '250-300'
from train
group by advertiser






