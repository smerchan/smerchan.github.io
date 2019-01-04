---
layout: post
type: blog
title: <center>Machine Learning Glossary</center>
comments: true
mathjax: true
---

## Introduction
The goal of this post is to briefly explain popular (and unpopular) concepts in Machine Learning, the idea for which stemmed from my travails for finding good quality explanations of various Machine Learning concepts on the web. Unlike similar posts on the web, here you'll also find links to good quality resources and to related concepts for more wholistic understanding. Hopefully, this post would be helpful to the people who are just starting in Machine Learning as well as to the people who need a quick refresher on some concepts. 

**Didn't find what you were looking for? Consider contributing by creating a pull request on this post [here](https://github.com/rishabhmisra/rishabhmisra.github.io/blob/master/_posts/2018-12-12-Machine-Learning-Glossary.md)**.

## Jump to<a name="top"></a>
[A](#A) . [B](#B) . [C](#C) . [D](#D) . [E](#E) . [F](#F) . [G](#G) . [H](#H) . [I](#I) . [J](#J) . [K](#K) . [L](#L) . [M](#M) . [N](#N) . [O](#O) . [P](#P) . [Q](#Q) . [R](#R) . [S](#S) . [T](#T) . [U](#U) . [V](#V) . [W](#W) . [X](#X) . [Y](#Y) . [Z](#Z)

## A<a name="A"></a>
* **AUC**<a name="AUC"></a>: AUC is the **A**rea **U**nder the Receiver Operating Characteristic (ROC) **C**urve. ROC curve is obtained by varying the classification threshold of a binary classifier and plotting the true positive rate ([TPR](#TPR)) against the false positive rate ([FPR](#FPR)) at each threhold. It is a popular classification performance metric and has several nice properties like being independent of decision threshold, being robust to class imbalance in data and so on.
  * Useful links: [Video Explanation of AUC](https://www.youtube.com/watch?v=OAl6eAyP-yo) \| [Probabilistic interpretation of AUC](https://www.alexejgossmann.com/auc/)
  
[Back to Top](#top)

## B<a name="B"></a>
* **Bagging**<a name="Bagging"></a>: Bagging is a procedure that produces several different training sets of the same size with replacement and then trains a machine learning model for each set. The predictions are produced by taking majority vote in a [classification](#Classification) task and by averaging in a [regression](#Regression) task. Bagging helps in reducing variance from models.
  * Also see: [Random Forest](#RF)
  * Useful links: [Video explanation by Udacity](https://www.youtube.com/watch?v=2Mg8QD0F1dQ) \| [Blog post on Medium](https://medium.com/@harishkandan95/bagging-the-skill-of-bagging-bootstrap-aggregating-83c18dcabdf1)
* **Bias Variance Trade-off**<a name="bias-variance"></a>: Bias is the difference between the average prediction of a model and the correct value which we are trying to predict. Variance is the variability of model prediction for a given data point because of its sensitivity to small fluctuations in the training set. If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.
  * Useful links: [Video explanation by Trevor Hastie](https://www.youtube.com/watch?v=VusKAosxxyk) \| [Blog post by towardsdatascience](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
* **Boosting**<a name="Boosting"></a>: Boosting is an ensemble method for improving the model predictions of any given learning algorithm. The idea is to train weak learners sequentially, each trying to correct its predecessor, to build strong learners. A weak learner is defined to be a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification.
  * Also see: [Bagging](#Bagging)
  * Useful links: [Lecture by Patrick Winston](https://www.youtube.com/watch?v=UHBmv7qCey4) \| [Boosting wiki](https://en.wikipedia.org/wiki/Boosting_(machine_learning))

[Back to Top](#top)

## C<a name="C"></a>
* **Classification**<a name="Classification"></a>: Classification is the problem of identifying to which of a set of categories a new observation belongs, on the basis of a training set of data containing observations whose category membership is known.
  * Also see: [Boosting](#Boosting) \| [Decision Trees](#DT) \| [K-Nearest Neighbor](#KNN) \| [Logistic Regression](#LoR) \| [Random Forest](#RF) \| [Naive Bayes Classifier](#NBC)
  * Useful links: [Classification Wiki](https://en.wikipedia.org/wiki/Statistical_classification)
* **Curse of Dimensionality**<a name="COD"></a>: In a model, as the number of features or dimensions grows, the amount of data needed to make the model generalizable with good performance grows exponentially, which unnecessarily increases storage space and processing time for a modeling algorithm. In this sense, value added by an additional dimension becomes much smaller compared to overhead it adds to the algorithm.
  * Also see: [Dimensionality Reduction](#DR)
  * Useful links: [Video explanation by Trevor Hastie](https://www.youtube.com/watch?v=UvxHOkYQl8g) \| [Elaborate post on Medium](https://medium.freecodecamp.org/the-curse-of-dimensionality-how-we-can-save-big-data-from-itself-d9fa0f872335)
  
[Back to Top](#top)

## D<a name="D"></a>
* **Decision Trees**<a name="DT"></a>:
* **Dimensionality Reduction**<a name="DR"></a>: The goal of dimensionality reduction methods is to find a low-dimensional representation of the data that retains as much information as possible. This low-dimensional data representation in turn helps in fighting the [Curse of Dimensionality](#COD).
  * Also see: [Principle Component Analysis](#PCA)
  * Useful links: [Video Explanation by Robert Tibshirani](https://www.youtube.com/watch?v=QlyROnAjnEk) \| [Blog post from towardsdatascience](https://towardsdatascience.com/https-medium-com-abdullatif-h-dimensionality-reduction-for-dummies-part-1-a8c9ec7b7e79)
  
[Back to Top](#top)

## E<a name="E"></a>
  
[Back to Top](#top)

## F<a name="F"></a>
* **False Positive Rate**<a name="FPR"></a>: The false positive rate is calculated as the ratio between the number of negative events wrongly categorized as positive (false positives) and the total number of actual negative events (regardless of classification).
  * Useful links: [False Positive Rate Wiki](https://en.wikipedia.org/wiki/False_positive_rate)
  
[Back to Top](#top)

## G<a name="G"></a>
  
[Back to Top](#top)

## H<a name="H"></a>
  
[Back to Top](#top)

## I<a name="I"></a>
  
[Back to Top](#top)

## J<a name="J"></a>
  
[Back to Top](#top)

## K<a name="K"></a>
* **K-Nearest Neighbor**<a name="KNN"></a>: KNN is essentially a classification technique that finds the ($K$) data points in the training data which are most similar to an unseen data point, and takes majority vote to make classifications. KNN is a non-parametric method which means that it does not make any assumptions on the underlying data distribution. Performance of KNN methods depend on the data representation and the definition of closeness/similarity.
  * Useful links: [Video explanation by Trevor Hastie](https://www.youtube.com/watch?v=vVj2itVNku4) \| [Blog post on Medium](https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7)
  
[Back to Top](#top)

## L<a name="L"></a>
* **Linear Regression**<a name="LiR"></a>: 
* **Logistic Regression**<a name="LoR"></a>: 

[Back to Top](#top)

## M<a name="M"></a>
  
[Back to Top](#top)

## N<a name="N"></a>
* **Naive Bayes Classifier**<a name="NBC"></a>: Naive Bayes Classifier is based on [Bayes’ Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). It assumes that the presence of a particular feature in a class is unrelated with the presence of any other feature and they all independently contribute towards the class probability.
  * Useful links: [Video Explanation by Trevor Hastie](https://youtu.be/6FiNGTYAOAA?t=275) \| [Blog post by towardsdatascience](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c)

[Back to Top](#top)

## O<a name="O"></a>
  
[Back to Top](#top)

## P<a name="P"></a>
* **Precision**<a name="Precision"></a>: If we are given a set of instances, precision is the fraction of relevant instances (those correctly classified into a certain class $C$) among the retrieved instances (those belonging to a certain class $C$). A perfect precision score of 1.0 means that every result retrieved by a search was relevant, but says nothing about whether all relevant documents were retrieved.
  * Also see: [Recall](#Recall)
  * Useful links: [Blog post by towardsdatascience](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) \| [Precision and Recall Wiki](https://en.wikipedia.org/wiki/Precision_and_recall)
* **Principle Component Analysis**<a name="PCA"></a>: PCA is a statistical procedure that transforms a set of observations of possibly correlated variables into a set of observations with linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance and each succeeding component variance in decreasing order with the constraint that it is orthogonal to the preceding components. Utilizing only few components that capture most of the variance in data helps in fighting the [Curse of Dimensionality](#COD).
  * Useful links: [Video Explanation by Stanford Profs](https://www.youtube.com/watch?v=ipyxSYXgzjQ) \| [Online Lesson by Penn State University](https://onlinecourses.science.psu.edu/stat505/node/49/)
  
[Back to Top](#top)

## Q<a name="Q"></a>
  
[Back to Top](#top)

## R<a name="R"></a>
* **Random Forest**<a name="RF"></a>: 
* **Recall**<a name="Recall"></a>: If we are given a set of instances, recall is the fraction of relevant instances (belonging to a certain class $C$) that have been retrieved (or correctly classified in $C$) over the total number of relevant instances. A recall of 1.0 means that every item from class $C$ was labeled as belonging to class $C$, but does not say anything about other items that were incorrectly labeled as belonging to class $C$.
  * Also see: [Precision](#Precision)
  * Useful links: [Blog post by towardsdatascience](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) \| [Precision and Recall Wiki](https://en.wikipedia.org/wiki/Precision_and_recall)
* **Regression**<a name="Regression"></a>: Regression is the problem of approximating a mapping function ($f$) from input variables ($X$) to a continuous output variable ($y$), on the basis of a training set of data containing observations in the form of input-output pairs.
  * Also see: [Linear Regression](#LiR)
  * Useful links: [Video Explanation by Trevor Hastie](https://www.youtube.com/watch?v=WjyuiK5taS8)

[Back to Top](#top)

## S<a name="S"></a>
* **Sensitivity**<a name="Sensitivity"></a>: Same as [Recall](#Recall).
* **Specificity**<a name="Specificity"></a>: If we are given a set of instances, specificity measures the proportion of actual negatives (instances not belonging to a particular class) that are correctly identified as such (e.g., the percentage of healthy people who are correctly identified as not having the condition).
  * Useful links: [Specificity Wiki](https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Sensitivity).
* **Supervised Learning**<a name="SL"></a>: Supervised learning is a task of learning a function that can map an unseen input to an output as accurately as possible based on the example input-output pairs known as training data.
  * Also see: [Classification](#Classification) \| [Regression](#Regression)
  * Useful links: [Coursera Video Explanation](https://www.coursera.org/lecture/machine-learning/supervised-learning-1VkCb) \| [Supervised Learning Wiki](https://en.wikipedia.org/wiki/Supervised_learning)
* **Support Vector Machines**<a name="SVM"></a>:
  
[Back to Top](#top)

## T<a name="T"></a>
* **True Positive Rate**<a name="TPR"></a>: Same as [Recall](#Recall).
* **True Negative Rate**<a name="TNR"></a>: Same as [Specificity](#Specificity).
  
[Back to Top](#top)

## U<a name="U"></a>
* **Unsupervised Learning**<a name="UL"></a>: Unsupervised learning is the task of inferring patterns from data without having any reference to known, or labeled, outcomes. It is generally used for discovering underlying structure of the data.
  * Also see: [Principle Component Analysis](#PCA)
  * Useful links: [Blog post by Hackernoon](https://hackernoon.com/unsupervised-learning-demystified-4060eecedeaf) \| [Coursera Video Explanation](https://www.coursera.org/lecture/machine-learning/unsupervised-learning-olRZo) 
  
[Back to Top](#top)

## V<a name="V"></a>
  
[Back to Top](#top)

## W<a name="W"></a>
  
[Back to Top](#top)

## X<a name="X"></a>
  
[Back to Top](#top)

## Y<a name="Y"></a>
  
[Back to Top](#top)

## Z<a name="Z"></a>

[Back to Top](#top)


<center> <img src="https://hitcounter.pythonanywhere.com/count/tag.svg" alt="Hits"> </center>
