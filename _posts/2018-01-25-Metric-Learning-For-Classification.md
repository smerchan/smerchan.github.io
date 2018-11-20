---
layout: post
type: blog
title: <center>Metric Learning for Classification</center>
comments: true
mathjax: true
---

## Introduction
In machine learning, models typically fall into two categories: learning based and similarity based. Learning based models have certain parameter which are tuned on training data to minimize a certain loss using an appropriate optimization method. Whereas, similarity based models produce predictions on unseen data instances based on their similarity (or closeness) to already seen data instances. They are robust when there is label noise in the dataset (that is, correct labels could themseleves be wrong). However, success of how such methods perform depends primarily on how we quantify the similarity/closeness between two data instances and to some extent on how we represent data instances. Typical distance measures like Euclidean distance might not be able to capture the notion of similarity in different scenarios. In this post, the agenda is to discuss Metric Learning technique, which is a combination of learning based and similarity based technique, that aims at automatically learning a Mahanolobis distance metric under which data instances of the same class are more similar as compared to instances of other class.
