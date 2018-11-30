---
layout: post
type: blog
title: <center>Would this clothing product fit me?</center>
comments: true
mathjax: true
---

## Introduction
Online shopping is trending these days as it provides many benefits like convenience, large selection of products, great prices and so on. Inspite of these benefits, shopping clothes online is still tricky because it's difficult to gauge their fit as every brand has slightly different sizing convention. Thus, recommending appropriate catalog sizes of clothing products is critical for improving online clothing shopping experience and reducing return rates. In this post, first we frame the catalog size recommendation problem, then deep dive into the challenges of recommending catalog sizes that fit and then finally elaborate a Machine Learning model that tries to address those challenges to provide better recommendations.

## Background
With the growth of the online fashion industry and the wide size variations across different clothing products, automatically providing accurate and personalized fit guidance is worthy of interest. Many online retailers nowadays allow customers to provide fit feedback (e.g. `small`, `fit` or `large`) on the purchased catalog size during the product return process or when leaving reviews. This data contributes two important signals `catalog size purchased` and `fit feedback of customer on that size`, which can be used to learn customers' "fit preferences" using Machine Learning. Once we learn the "fit preferences", we can serve customers with recommendations of better fitting catalog sizes of products.

## Challenges
Before we talk about how we can serve catalog size recommendations, let's deep dive into the challenges to understand the complexity of the problem. Some of the challenges could be: 
* Customers' fit feedback reflects not only the objective match/mismatch between a product's true size and a customer's measurements, but also depends on other subjective characteristics of a product's style and properties.
* Customers' fit feedback is unevenly distributed as most transactions are reported as `fit`, so it is difficult to learn when the purchase would be "unfit".
