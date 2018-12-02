---
layout: post
type: blog
title: <center>Would this clothing product fit me?</center>
comments: true
mathjax: true
---

## Introduction
Online shopping is trending these days as it provides many benefits like convenience, large selection of products, great prices and so on. Trailing on this trend, online fashion industry has also seen tremendous growth. However, shopping clothes online is still tricky because it is hard to gauge their fit given the wide size variations across different clothing brands. Thus, automatically providing accurate and personalized fit guidance is critical for improving online shopping experience and reducing product return rates. 

This is an explainatory post on my [recent RecSys paper](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18e.pdf). The outline of this post is like following: i) We frame the catalog size recommendation problem; ii) Briefly explain the data we can use to tackle the problem; iii) Briefly explain a simple machine learning model developed by Amazon last year to tackle this problem; iv) Deep dive into the challenges overlooked by the aforementioned model; v) Elaborate a machine learning model that tries to address those challenges to further improve recommendations.

## Prerequesite concepts
Following concepts are required to understand this post thoroughly. Though, I'll briefly explain some of them, the corresponding links would provide in-depth knowledge.
* [Latent Factor Model](http://www.ideal.ece.utexas.edu/seminar/LatentFactorModels.pdf)
* [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
* [Ordinal Regression](http://www.norusis.com/pdf/ASPC_v13.pdf)
* [Hinge Loss](https://en.wikipedia.org/wiki/Hinge_loss)

## Catalog Size Recommendation Problem
Many online retailers nowadays allow customers to provide fit feedback (e.g. `small`, `fit` or `large`) on the purchased product size during the product return process or when leaving reviews. For distinction, we will refer to the product (e.g. a Northface Jacket) as parent product and the different size variations (e.g. a Medium sized Northface Jacket) as child products. So, each purchase can be represented as a triple of the form (customer, child product, fit feedback), which contributes two important signals: `product size purchased` and `fit feedback of customer on that size`. Given this type of data, the goal of catalog size recommendation is to learn customers' fit preferences and products' sizing related properties so that customers can be served with recommendations of better fitting product sizes.

## Data
For the purpose of this research, we collected data from [ModCloth](https://www.modcloth.com/) and [RentTheRunWay](https://www.renttherunway.com/) websites. These datasets contain self-reported fit feedback from customers as well as other side information like reviews, ratings, product categories, catalog sizes, customers’ measurements (etc.). For more details and to get access to the data, please head over to the [dataset page](https://www.kaggle.com/rmisra/clothing-fit-dataset-for-size-recommendation/home) on Kaggle. 

## A [simple model](http://delivery.acm.org/10.1145/3110000/3109891/p243-sembium.pdf?ip=73.53.61.10&id=3109891&acc=OA&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E538B033CF25F0137&__acm__=1543720447_839bf53830210baf4493f6bd6457b777) developed by Amazon
For every customer and product, we consider a latent variable which denotes their true sizes. Let $u_c$ denote the true size for customer $c$ and $v_p$ be the true size for product $p$. We can learn these variables using past transactions in which customers have provided their fit feedback. Once we have the true sizes, we recommend the child product $p$ whose true size $v_p$ is closest to the customer’s true size $u_c$.

## Challenges
Before we talk about how we can serve catalog size recommendations, let's understand the complexity of the problem through the challenges it poses. Some of the challenges could be: 
* Customers' fit feedback reflects not only the objective match/mismatch between a product's size and a customer's measurements, but also depends on other subjective characteristics of a product's style and properties.
* Customers' fit feedback is unevenly distributed as most transactions are reported as `fit`, so it is difficult to learn when the purchase would be `unfit`.
