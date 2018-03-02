---
layout: post
type: blog
title: Maximum Likelihood Estimates - Motivation for EM algorithm
comments: true
mathjax: true
---

Once we have obtained a dataset and designed a model, our next task is to find a way to estimate the parameters of the model based on the dataset we have, so that we can make predictions on unseen data. In this post, we will learn about how we can learn the parameters of the model using Maximum Likelihood approach which has a very simple premise: find parameters that maximize the likelihood of the observed data. Through that, I would motivate the Expectation-Maximization (EM) algorithm which is considered to be an important tool in statistical analysis.

Now, let's understand Maximum Likelihood approach in the context of Logistic Regression. Suppose, we have a dataset where we denote predictor variables with $X \in R^d$ and the target variable with $Y \in \{0,1\}$. The model could be depicted graphically as following:

<center>
<img src="/images/mle/logistic_model.JPG" width="600" height ="300"/>
</center>

As we know, prediction probability of target variable in logistic regression is given by a sigmoid function like following:
<center>
$P(Y = 1 | X = x) = \sigma(w.x) = \frac{1}{1 + exp(-w.x)}$
</center>
Based on whether 1 or 0 has more probability of occuring based on the predictor variables, we output our prediction. But, how do we appropriate $w$ such that the error is minimized?

To that end, we will use the Maximum Likelihood approach where we'll try to find the $w$ which maximizes the likelihood of the observed data. For mathematical convenience, we'll consider try to maximize the log of the likelihood. Likelihood of data, parameterized by $z$, can be written as: 

<center>
$L(w) = log P(\text{data}) = log \Pi_{i=1}^N P(Y = y_i | X = x_i)$
</center>
<br/>
<center>
$L(w) = \sum_{i=1}^N log  P(Y = y_i | X = x_i) = \sum_{i=1}^N log\left[\sigma(w.x_i)^{y_i} . \sigma(-w.x_i)^{1 - y_i} \right]$
</center>
where we've written $P(Y = y_i | X = x_i)$ in a general form.

<center>
$L(w) = \sum_{i=1}^N \left[ y_i. log\sigma(w.x_i) + (1 - y_i). log\sigma(-w.x_i) \right]$ --- (A)
</center>

At this point, we should note that log likelihood $L(w)$ breaks down conveniently into per-instance form. Since, there's no coupling between the parameters, optimization can be done easily and we'll see later why this is a good thing. Since $L(w)$ is a function of $w$, we don't have any closed form solution to equation (A). Thus, we would have to use iterative optimization methods like gradient ascent or newton's method to find $w$. An update for gradient ascent method would look like:
<center>
$w = w + \eta.\left( \sum_{i=1}^N [y_i - \sigma(w.x_i)].x_i \right)$ --- (B)
</center>
where $\eta$ is an appropriate learning rate. We repeat (B) until convergence. The final value of $w$ we opt for is called maximum likelihood estimate.

While going through above steps, I skipped one detail about the model. If you notice, all the predictor variables in the model are observed. However, there can be situations, where we have latent (unobserved) predictor variables in the model. One great example for this kind of model is the [Hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model). Briefly, in the hidden Markov model, the actual state is not directly visible, but the output, dependent on the state, is visible. This model has applications in speech, handwriting, gesture recognition, part-of-speech tagging, musical score following and other areas.

Estimating model parameters in this case gets a little tricky for the reasons I'll describe following.
