
# Facebook-Sentiment-Analysis

### Jeong eun (Hailey) Lee, jlee562@jhu.edu 
### Alex Ahn 
### Jae Goan Park 


## Note: Please view the "Project Write-Up" PDF for more in-depth analysis, findings, and conclusion. 
Also, to run any of our Python programs (the models in this folder and the scraper engines), you will need to have the following libraries installed: tweepy, pquery, pycparser, beautifulsoup4, lxml, Flask, Flask-Tweepy

## Overview
Sentiment Analysis is more than figuring out how people feel about in the social media. With sophisticated analysis on how people react to certain topics, sentiment analysis can predict the following: campaign success, marketing strategy, product messaging, customer service, and stock market price. We decided to take advantages of recent extension of reactions made by Facebook and do sentiment analysis on how people react differently for different posts. Our dataset consist of all posts from “Opposing View” from Facebook Public page from August 2016 to April 2017. 


## Procedure
We collected about 3 MB data which includes, the actual text, number of shares, comments, likes, reactions, and number of all reactions, by using public Facebook API. After pre-processing this data by using stemming, stoplists to vectorize the documents, we performed LDA ( Topic Modelling algorithm), generated 10 topic vectors, and assigned the most relevant topic label. With the generated 10 topic vectors, we performed PCA to visualize the distribution, created a radar chart to see the distribution of emotion to each topic, and also two correlational matrices to visualize the relationship between topics. 

