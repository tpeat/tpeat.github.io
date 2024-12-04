---
layout: post
title: Concluded Automated Algorithm Design research with VIP ✅
date: 2023-12-01 16:11:00-0400
inline: false
related_posts: false
---

As a student at Georgia Tech, I had the unique opportunity to participate in the Vertically Integrated Projects (VIP) program, specifically working on the Evolutionary Model Architecture Design Engine (EMADE) team. This project combined evolutionary algorithms with neural network architecture search - essentially teaching computers to design their own AI models. Over three semesters, I went from barely understanding genetic programming to helping develop novel reinforcement learning approaches for portfolio optimization. Let me take you through that journey.

## Fall 2022: Learning the Fundamentals

My first semester started with the basics - implementing genetic algorithms to solve classic computer science problems like the N-Queens puzzle and One-Max optimization. While these seemed simple, they taught me crucial concepts about how evolutionary algorithms work: population dynamics, fitness functions, mutation, and crossover operations.

The semester's main challenge was tackling the Titanic dataset from Kaggle using genetic programming. Our team had to evolve decision trees that could predict passenger survival while optimizing for both false positive and false negative rates. This introduced me to multi-objective optimization and the concept of Pareto frontiers - where you can't improve one metric without hurting another. I spent countless hours tuning primitives (the basic building blocks our trees could use) and selection strategies to improve our results.

## Spring 2023: Deep Reinforcement Learning for Portfolio Optimization

The following semester, I joined the stocks team where we tackled a more ambitious challenge: applying deep reinforcement learning to portfolio optimization. Our goal was to evolve trading strategies that could maximize returns while managing risk. We developed a system using LSTM networks to process market data from Yahoo Finance and output portfolio allocation weights, optimized using the Sharpe Ratio as our performance metric. 

This semester, I spent most of my time shadowing Tyler Feeny, either through his notebook history, git commits, or asking thousands of questions. Through our conversations, I was able to conceptualize the bigger problem we were trying to solve and visualize the pathway to get there.

Integrating this framework into EMADE proved challenging since the existing system wasn't designed for reinforcement learning data, which doesn't have explicit labels but rather acts on unlabeled experiences. We still needed a way to give reward so we calculated the sharpe ratio based on the input data and our models output, reweighted distribution of stocks. The solution was to pass the same data into the (X, Y) data pairs in emade so that the Sharpe Ratio custom loss could act on the Y data and our prediction `model(X)` to compute expected reward against the original scenario. Given this successful integration, we could not only evolve hyperparameters but also implement learnable feature selectors to identify important market signals - particularly valuable in finance where there are numerous predefined metrics for finding alpha.

While we made significant progress tuning LSTM parameters like depth and embedding dimensions, we realized the potential of evolving the neural architecture itself. This insight led us to focus on neural architecture search in our final semester.

## Fall 2023: Advancing Neural Architecture Search

Building on our previous work, I focused on expanding EMADE's capabilities for neural architecture search. Drawing inspiration from successful architectures like GoogLeNet and ResNet, I implemented support for inception modules and sophisticated skip connections. The challenge lay in finding ways to effectively encode these complex structures so they could evolve naturally.

My major contribution was developing a system for handling branching and recombination in neural networks. This innovation allowed EMADE to evolve architectures where data flow could split, process in parallel through different operations, and recombine - similar to modern architectures like ResNext or InceptionNet. We began viewing the model as a DAG of modules instead of a sequential linked list of modules. For mutations, we experimented with inserting new modules into the DAG, removing stale modules (ranked by activation magnitude), or reshuffling the layers. These proved increasingly difficult for preserving spatial dimensions and autograd flow. I have never spent more hours inspecting tensor shapes than during this semester. However, I became a master of `x.reshape`, `x.view` and `x.permute`. Ultimately, the easiest solution was to force convolutions to set `padding=True` to preserve shape and use a consisent embedding dimension so that all tensors fit together. 

Another tricky debugging issue was GPU hogging. Due to the distributed nature of EMADE, often subprocesses would lock onto a GPU and never give it up, even though it was done with its training. The subprocesses were often so deep in the call stack, it was impossible to see their `stdout`. The cause of most of the deadlock issues were OOM errors, so the challenges subsided once we ran all experiments on the A100-80GB GPUs. 

The progression from simple genetic algorithms to sophisticated neural architecture search taught me valuable lessons about both evolutionary computation and deep learning. More importantly, it showed me that the future of machine learning might not lie in hand-crafted architectures, but in systems that can evolve and optimize themselves.

I also had the pleasure of working with some incredibly talented researchers. Tyler Feeny first inspired me to take ownership on the Stocks team; Elijah Nicpon brought a light-hearted perspective to deep learning challenges; Brandon Conner's leadership in Fall 2023 made our work possible; and Tristan Thakur and Hugh Westin showed remarkable dedication in ensuring the future success of the NAS team.

This VIP experience went beyond technical achievements. It taught me how to tackle complex research problems, work with distributed systems, and collaborate on long-term projects. These skills - problem-solving, system design, and teamwork - will serve me well throughout my career, regardless of the specific technologies involved.
