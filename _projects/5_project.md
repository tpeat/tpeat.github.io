---
layout: page
title: Evolutionary Neural Architecture Search
description: 
img: assets/img/transformer.jpg
importance: 3
category: fun
related_publications: true
---

I began my journey in the world of genetic programming (GP) through the Vertically Integrated Project (VIP) called Automated Algorithm Design (AAD). I enrolled in the VIP to fufill my capstone requirement, but also because the team seemed to be doing the most general machine learning work. Other VIP teams were siloed to specific applications, like self driving cars or generating art. I felt as though this project team would have the most opportunity to expose myself to a variety of topics. Boy was I right.

To give further context, at the time of joining (Fall 2022), I was interested in time series prediction, aka stock market prediciton and was attending Blockchain and Trading club meetings. The AAD stock market subteam had just published a paper with an algorithm that produced 37% returns per annum by selecting different technical indicators to prepocess data before feeding to price point prediction models (up or down). I was determined to take this to the next level.

My first semester of AAD, I learned the basics of GP, ML, and evolutionary optimization. The idea of evolutionary optimization resembles survival of the fittest but for machine learning algorithms. First, encode an algorithm in a manner that it can be represented like a DNA sequences, where you can combine and mutates 'strands'. Building tree's from a set of predefined primitives is the typical approach for this encoding process. Then trees can be combined by chosing a node in `tree_1` and a node in `tree_2` to split each tree by and randomly combine them again. It is trivial to implement mutations in trees by changing one or more nodes in the tree. The simplest set of primitives that have the power to build function might be `{addition, subtraction}`. 

Then input is passed through the trees to produce an output which is scored on one or more objective. If we were trying to fit a line we might use standard MSE or for classification use F1 score. If we cared about efficient tree structures in terms of space or operators, we might try to minimize number of operators or depth of the tree.

Lastly, the best scoring individuals are mated together to produce a number of similar but slightly different offspring. Randomly, some offspring will be mutated, and the process of evolution will continue.

The assignments for the course were extremely open ended: Find the best tree that performs binary classification on the classic kaggle Titanic dataset. This left me a lot of opportunity to experiment with different algorithms and search for new ones. I built my own primitives, drawing knowledge from well known ML models like XGBoost.

My second semester in AAD looked very different; I joined the stocks subteam and immediately began working on Deep Learning Reinforcment Learning for Portfolio Optimization.