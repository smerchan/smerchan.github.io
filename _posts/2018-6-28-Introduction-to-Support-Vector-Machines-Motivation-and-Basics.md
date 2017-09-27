---
layout: post
type: blog
title: Introduction to Support Vector Machines
excerpt: Motivation and Basics
comments: true
mathjax: true
---

In this post, you will learn about the basics of Support Vector Machines (SVM), which is a well-regarded supervised machine learning algorithm. This technique needs to be in everyone's tool-bag especially people who aspire to be a data scientist one day. Since there's a lot to learn about, I'll introduce SVM to you across two posts so that you can have a coffee break in between :)

First, let us try to understand how SVM works in the context of a binary classification problem. In a binary classification problem, our data belong to two classes and we try to find a decision boundary that splits the data into those two classes while making minimum mistakes. Consider the diagram below which represents our (hypothetical) data on a 2-d plane. As we can see, the data is divided into two classes: Pluses and Stars.

Note: For the sake of simplicity, we'll only consider linearly separable data for now and learn about not linearly separable cases later on.

<center>
<img src="/images/svm/data_points.PNG" width="400" height ="400"/>
</center>

The goal of SVM, like any classification algorithm, is to find a decision boundary which splits the data into two classes. However, there could be many possible decision boundaries to achieve this purpose like shown below. Which one should we consider?

<center>
<img src="/images/svm/decision_boundaries.PNG" width="400" height ="400"/>
</center>

The yellow and the black decision boundaries do not seem to be a good choice. Why, you ask? This is simply because they might not generalize well to new data points as each of them is awfully close to one of the classes. In this sense, the blue line seems to be a good candidate as it is far away from both classes. Hence, by extending this chain of thought, we can say that an ideal decision boundary would be a line that is at a maximum distance from any data point. In this sense, if we think of the decision boundary as a road, we want that road to be as wide as possible. This is exactly what SVM aims to do.

Phew. Now that we understand what SVM aims to do, our next step is to understand how it finds this decision boundary. So, let's start from scratch with the help of the following diagram.

<center>
<img src="/images/svm/threshold_equation.PNG" width="400" height ="400"/>
</center>

First, we will derive the equation of the decision boundary in terms of the data points. To that end, let us suppose we already have a decision boundary (blue line in above diagram) and two unknown points which we have to classify. We represent these points as vectors $ \vec{u}$ and $ \vec{v}$ in the 2-d space. We also introduce a vector $ \vec{w}$ which we assume is perpendicular to the decision boundary. Now, we project $ \vec{u}$ and $ \vec{v}$ in the direction of $ \vec{w}$ and check whether the projected vector is on the left or right side of the decision boundary based on some threshold $ c$.

Mathematically, we say that a data point $ \vec{x}$ is on the right side of decision boundary (that is, in the Star class) if $ \vec{w} . \vec{x} \ge c$ else it is in the plus class. This means that the equation of the hyperplane (line in case of 2-d) that separates two classes, in terms of an arbitrary data point $ \vec{x}$, is following:

$$ \vec{w} . \vec{x} + b = 0$$ 
where $ b = -c$

Now we have the equation of our decision boundary but it is not yet immediately clear how it would help us in the maximizing its distance from the data points of both the classes. To that end, we would employ a trick which goes as follows. Usually, in a binary classification problem, the labels of data samples are + 1 or -1. Thus, it would be more convenient for us if our decision rule (i.e. $ \vec{w} . \vec{x} + b$) outputs quantity greater than or equal to +1 for all the data points belonging to star class and quantity less than or equal to -1 for all the data points belonging to plus class.

Mathematically, $ \vec{x}$ should belong to class Star if $ \vec{w} . \vec{x} + b \ge 1$ and $ \vec{x}$ should belong to class Plus if $ \vec{w} . \vec{x} + b \le -1$ or equivalently, we can write 

$$ y_{i} (\vec{w} . \vec{x}_{i} + b) \ge 1$$ 
for each point $ \vec{x}_{i}$ where we are considering $ y_{i}$ equal to -1 for plus class and equal to +1 for star class.

These two rules correspond to the dotted lines in the following diagram and the decision boundary is parallel and at equal distance from both. As we can see, the points closest to the decision boundary (on either side) get to dictate its position. Now, since the decision boundary has to be at a maximum distance from the data points, we have to maximize the distance $ d$ between the dotted lines. By the way, these dotted lines are called support vectors.
