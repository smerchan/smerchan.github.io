---
layout: post
type: blog
title: <center>Would this clothing product fit me?</center>
comments: true
mathjax: true
---

## Introduction
Online shopping is trending these days as it provides many benefits like convenience, large selection of products, great prices and so on. Trailing on this trend, online fashion industry has also seen tremendous growth. However, shopping clothes online is still tricky because it is hard to gauge their fit given the wide size variations across different clothing brands. Thus, automatically providing accurate and personalized fit guidance is critical for improving online shopping experience and reducing product return rates. 

This is an explainatory post on my [recent RecSys paper](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18e.pdf). The outline of this post is like following: i) We frame the catalog size recommendation problem; ii) Briefly explain a simple machine learning model developed by Amazon last year to tackle this problem; iii) Deep dive into the challenges overlooked by the aforementioned model; iv) Briefly explain the data we can use to tackle the problem; v) Elaborate a machine learning model that tries to address those challenges to further improve recommendations; vi) Highlight the results; and vii) Some future directions to work on. 

### Prerequesite concepts
Following concepts are required to understand this post thoroughly. Though, I'll briefly explain some of them, the corresponding links would provide in-depth knowledge.
* [Latent Factor Model](http://www.ideal.ece.utexas.edu/seminar/LatentFactorModels.pdf)
* [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
* [Ordinal Regression](http://www.norusis.com/pdf/ASPC_v13.pdf)
* [Hinge Loss](https://en.wikipedia.org/wiki/Hinge_loss)

## Catalog Size Recommendation Problem
Many online retailers nowadays allow customers to provide fit feedback (e.g. `small`, `fit` or `large`) on the purchased product size during the product return process or when leaving reviews. For distinction, we will refer to the product (e.g. a Northface Jacket) as parent product and the different size variations (e.g. a Medium sized Northface Jacket) as child products. So, each purchase can be represented as a triple of the form (customer, child product, fit feedback), which contributes two important signals: `product size purchased` and `fit feedback of customer on that size`. Given this type of data, the goal of catalog size recommendation is to learn customers' fit preferences and products' sizing related properties so that customers can be served with recommendations of better fitting product sizes.

## A simple model developed by Amazon
Researchers from Amazon India developed [a simple model](http://delivery.acm.org/10.1145/3110000/3109891/p243-sembium.pdf?ip=73.53.61.10&id=3109891&acc=OA&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E538B033CF25F0137&__acm__=1543720447_839bf53830210baf4493f6bd6457b777) last year to recommend shoe sizes to customers. For every customer and product, they consider a latent variable which denotes their true size. Let $u_c$ denote the true size for customer $c$ and $v_p$ be the true size for product $p$. They learn these variables using past transactions in which customers have provided their fit feedback. For a product, the true size is different from its catalog size because of the size variations across different brands. The learned true sizes bring the sizing of all the products on the same scale which makes it easier to gauge their fit.

Intuitively, if there's a transaction ($c$, $p$, `fit`), then the true sizes $u_c$ and $v_p$ must be close, that is, \|$u_c - v_p$\| must be small. On the other hand, if there's a transaction ($c$, $p$, `small`), then the customer's true size $u_c$ must be much larger than the product's true size $v_p$, or equivalently, \|$u_c - v_p$\| must be large. Similarly, for the `large` case, $v_p - u_c$ must be large. They assign a fit score to each transaction $t$ as: $f_w(t)=w(u_{t_c} - v_{t_p})$, where $w$ is the non-negative weight parameter, and pose the machine learning objective as: "Given past customer transactions $D$, actual catalog sizes for products, and loss function $L(y_{t} , f_w(t))$ for transactions $t \in D$, compute true sizes {$u_{t_c}$} for customers and {$v_{t_p}$} for child products such that $L = \sum_{t \in D} L(y_t , f_w(t))$ is minimized."

For the experimentation, they considered Hinge Loss as the loss function. Once true sizes are learned, they use them as features in standard classifiers like Logistic Regression Classifier and Random Forest Classifier to produce the final fit outcome for recommendation.

## Challenges
Although the aforementioned model works, it does not address some key challenges mentioned following: 
* Customers' fit feedback reflects not only the objective match/mismatch between a product's size and a customer's measurements, but also depends on other subjective characteristics of a product's style and properties.
  * For example, consider a Jacket and a Wet Suit. Usually, customers prefer the fitting of Jacket to be a little bit loose whereas fittinf of Wet Suit to be more form fitting. Furthermore, for each customers, fit preferences would also vary across different product categories. Hence, capturing true sizes of customers and products with a single latent variable might not be sufficient. 

* Customers' fit feedback is unevenly distributed as most transactions are reported as `fit`, so it is difficult to learn when the purchase would be `unfit`.
  * Standard classifiers are not capable of handling the label imbalance issue and results in biased classification, i.e. in this case, `small` and `large` classes will have poor estimation rate.
  

## Datasets
Since the dataset used in Amazon's study was proprietary, for this research we collected datasets from [ModCloth](https://www.modcloth.com/) and [RentTheRunWay](https://www.renttherunway.com/) websites. These datasets contain self-reported fit feedback from customers as well as other side information like reviews, ratings, product categories, catalog sizes, customers’ measurements etc. For more details and to access the data, please head over to the [dataset page](https://www.kaggle.com/rmisra/clothing-fit-dataset-for-size-recommendation/home) on Kaggle.

## A New Approach
We tackle the aforementioned challenges in the following ways: First, unlike previous work which focuses on recovering "true" sizes, we develop a new model to factorize the semantics of customers’ fit feedback, so that representations can capture customers’ fit preferences on various product aspects (like shoulders, waist etc.). Second, using a heuristic we sample good representations from each class and project them to a metric space to address label imbalance issues. The overview of framework can be understood from the following diagram:

<center>
<img src="/images/projects/recsys18_framework.jpg" width="85%" height ="500"/>
</center>

We explain our approach in detail in following subsections.

### Learning Fit Semantics
We quantify the fitness adopting latent factor model formulation as:

<center>
<img src="/images/fit/fit_score_eq.png" width="60%" height ="100"/>
</center>

### Handling Label Imbalance
