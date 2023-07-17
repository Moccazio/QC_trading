# Mixture Model Trading on Quantconnect

This strategy is a test of an gmm inverse VAR portfolio allocation algorithm verse some common benchmarks. 

The symbol universe consisted of the following ETFs:

_"SPY", "QQQ", "DIA", "TLT", "GLD", "EFA", "EEM", 'BND', 'VNQ'_

- All the algorithms start from `2008-05-01` and run until `2023-07-01`.
- Rebalance daily.
- Target weights had a 5% corridor or +/- 2.5% (to minimize transaction costs). 
- Lookback is either 252 days or 60 days. 

The benchmark algorithms are the equal weight and 60/40 SPY/BND algorithm. 

I tested 2 Historical VAR models. 
One with a lookback of 252 days and the other with a lookback 60 days. 

I tested 2 GMM VAR models.
One with a lookback of 252 days and the other 60 days. 
I restricted the model to 2 components.

