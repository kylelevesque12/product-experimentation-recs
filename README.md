Will build upon this as the project progresses. For now, not much to comment on 

## Positive Interaction Definition
For this project, I defined a positive recommendation as a rating of 4.0 stars or higher. 

## Data Source
This project uses the 'latest-small' dataset from GroupLens researchers at the University of Minnesota. Contained within this dataset are anonymized user ratings, tagging activity, and movie metadata from the MovieLens movie recommendation service. While the raw dataset is not included in this repository, it can be downloaded from the official GroupLens MovieLens dataset page. Note that using this dataset requires following the GroupLens usage terms and citing the following in any publications: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19.
- https://grouplens.org/datasets/movielens/latest/
- https://doi.org/10.1145/2827872

## Methodology Notes
temporal user split: Train on user's earlier ratings and test on their future ratings. This approach better mimics the real design of this experiment, which is, given what we know about a user in the past, can we recommend things they liked later? This also avoids leakage from future interactions into training. Our entire goal is essentially to predict the future ratings, so using future interactions into training can produce misleading product results, as a real product would not know the future yet. 