---
layout: post
type: blog
title: <center>Machine Learning Glossary</center>
comments: true
mathjax: true
---

## Introduction
The goal of this post is to briefly explain popular (and unpopular) concepts in Machine Learning along with a collection of helpful links for further understanding. The idea for this post stemmed from my travails for finding good quality explanations of various Machine Learning concepts on the web. Hopefully, this post would be helpful to people who are just starting in Machine Learning as well as to people who need a quick refresher on some concepts. 

**Didn't find what you were looking for? Consider contributing by creating a pull request on this post [here](https://github.com/rishabhmisra/rishabhmisra.github.io/blob/master/_posts/2018-12-12-Machine-Learning-Glossary.md)**.

## Jump to<a name="top"></a>
[A](#A) . [B](#B) . [C](#C) . [D](#D) . [E](#E) . [F](#F) . [G](#G) . [H](#H) . [I](#I) . [J](#J) . [K](#K) . [L](#L) . [M](#M) . [N](#N) . [O](#O) . [P](#P) . [Q](#Q) . [R](#R) . [S](#S) . [T](#T) . [U](#U) . [V](#V) . [W](#W) . [X](#X) . [Y](#Y) . [Z](#Z)

## A<a name="A"></a>
* **AUC**<a name="AUC"></a>: AUC is the **A**rea **U**nder the Receiver Operating Characteristic (ROC) **C**urve. ROC curve is obtained by varying the classification threshold of a binary classifier and plotting the true positive rate ([TPR](#TPR)) against the false positive rate ([FPR](#FPR)) at each threhold. It is a popular classification performance metric and has several nice properties like being independent of decision threshold, being robust to class imbalance in data and so on.
  * Useful links: [Video Explanation of AUC](https://www.youtube.com/watch?v=OAl6eAyP-yo) \| [Probabilistic interpretation of AUC](https://www.alexejgossmann.com/auc/)
  
[Back to Top](#top)

## B<a name="B"></a>
* **Bias Variance Trade-off**<a name="bias-variance"></a>: Bias is the difference between the average prediction of a model and the correct value which we are trying to predict. Variance is the variability of model prediction for a given data point because of its sensitivity to small fluctuations in the training set. If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.
  * Useful links: [Video explanation by Trevor Hastie](https://www.youtube.com/watch?v=VusKAosxxyk) \| [Blog post by towardsdatascience](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)

## C<a name="C"></a>
* **Curse of Dimensionality**<a name="COD"></a>: In a model, as the number of features or dimensions grows, the amount of data needed to make the model generalizable with good performance grows exponentially, which unnecessarily increases storage space and processing time for a modeling algorithm. In this sense, value added by an additional dimension becomes much smaller compared to overhead it adds to the algorithm.
  * Also see: [Dimensionality Reduction](#DR)
  * Useful links: [Video explanation by Trevor Hastie](https://www.youtube.com/watch?v=UvxHOkYQl8g) \| [Elaborate post on Medium](https://medium.freecodecamp.org/the-curse-of-dimensionality-how-we-can-save-big-data-from-itself-d9fa0f872335)

## D<a name="D"></a>
* **Dimensionality Reduction**<a name="DR"></a>: The goal of dimensionality reduction methods is to find a low-dimensional representation of the data that retains as much information as possible. This low-dimensional data representation in turn helps in fighting the [Curse of Dimensionality](#COD).
  * Also see: [Principle Component Analysis](#PCA)
  * Useful links: [Video Explanation by Robert Tibshirani](https://www.youtube.com/watch?v=QlyROnAjnEk) \| [Blog post from towardsdatascience](https://towardsdatascience.com/https-medium-com-abdullatif-h-dimensionality-reduction-for-dummies-part-1-a8c9ec7b7e79)

## E<a name="E"></a>

## F<a name="F"></a>
* **False Positive Rate**<a name="FPR"></a>: The false positive rate is calculated as the ratio between the number of negative events wrongly categorized as positive (false positives) and the total number of actual negative events (regardless of classification).
  * Useful links: [False Positive Rate Wiki](https://en.wikipedia.org/wiki/False_positive_rate)

## G<a name="G"></a>

## H<a name="H"></a>

## I<a name="I"></a>

## J<a name="J"></a>

## K<a name="K"></a>
* **K-Nearest Neighbor**<a name="KNN"></a>: KNN is essentially a classification technique that finds the ($K$) data points in the training data which are most similar to an unseen data point, and takes majority vote to make classifications. KNN is a non-parametric method which means that it does not make any assumptions on the underlying data distribution. Performance of KNN methods depend on the data representation and the definition of closeness/similarity.
  * Useful links: [Video explanation by Trevor Hastie](https://www.youtube.com/watch?v=vVj2itVNku4) \| [Blog post on Medium](https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7)

## L<a name="L"></a>

## M<a name="M"></a>

## N<a name="N"></a>

## O<a name="O"></a>

## P<a name="P"></a>
* **Precision**<a name="Precision"></a>: If we are given a set of instances, precision is the fraction of relevant instances (those correctly classified into a certain class $C$) among the retrieved instances (those belonging to a certain class $C$). A perfect precision score of 1.0 means that every result retrieved by a search was relevant, but says nothing about whether all relevant documents were retrieved.
  * Also see: [Recall](#Recall)
  * Useful links: [Blog post by towardsdatascience](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) \| [Precision and Recall Wiki](https://en.wikipedia.org/wiki/Precision_and_recall)
* **Principle Component Analysis**<a name="PCA"></a>:


## Q<a name="Q"></a>

## R<a name="R"></a>
* **Recall**<a name="Recall"></a>: If we are given a set of instances, recall is the fraction of relevant instances (belonging to a certain class $C$) that have been retrieved (or correctly classified in $C$) over the total number of relevant instances. A recall of 1.0 means that every item from class $C$ was labeled as belonging to class $C$, but does not say anything about other items that were incorrectly labeled as belonging to class $C$.
  * Also see: [Precision](#Precision)
  * Useful links: [Blog post by towardsdatascience](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) \| [Precision and Recall Wiki](https://en.wikipedia.org/wiki/Precision_and_recall)

## S<a name="S"></a>
* **Sensitivity**<a name="Sensitivity"></a>: Same as [Recall](#Recall).
* **Specificity**<a name="Specificity"></a>: If we are given a set of instances, specificity measures the proportion of actual negatives (instances not belonging to a particular class) that are correctly identified as such (e.g., the percentage of healthy people who are correctly identified as not having the condition).
  * Useful links: [Specificity Wiki](https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Sensitivity).
* **Supervised Learning**<a name="SL"></a>: Supervised learning is a task of learning a function that can map an unseen input to an output as accurately as possible based on the example input-output pairs known as training data.
  * Also see: [Unsupervised Learning](#UL)
  * Useful links: [Coursera Video Explanation](https://www.coursera.org/lecture/machine-learning/supervised-learning-1VkCb) \| [Supervised Learning Wiki](https://en.wikipedia.org/wiki/Supervised_learning) 

## T<a name="T"></a>
* **True Positive Rate**<a name="TPR"></a>: Same as [Recall](#Recall).
* **True Negative Rate**<a name="TNR"></a>: Same as [Specificity](#Specificity).

## U<a name="U"></a>
* **Unsupervised Learning**<a name="UL"></a>: Unsupervised learning is the task of inferring patterns from data without having any reference to known, or labeled, outcomes. It is generally used for discovering underlying structure of the data.
  * Also see: [Supervised Learning](#SL)
  * Useful links: [Blog post by Hackernoon](https://hackernoon.com/unsupervised-learning-demystified-4060eecedeaf) \| [Coursera Video Explanation](https://www.coursera.org/lecture/machine-learning/unsupervised-learning-olRZo) 

## V<a name="V"></a>

## W<a name="W"></a>

## X<a name="X"></a>

## Y<a name="Y"></a>

## Z<a name="Z"></a>

[Back to Top](#top)


<center> <img src="https://hitcounter.pythonanywhere.com/count/tag.svg" alt="Hits"> </center>
