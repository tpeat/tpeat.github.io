---
layout: post
title: Survey of Image genAI
date: 2023-12-01 21:01:00
description: Survey of some recent papers on image generation models
tags: cv
categories:
thumbnail: assets/img/9.jpg
---

Last November, I attended a talk by Dr. Yang Song through ML@GT seminar series on Consistency Models. Inspired by his clear delivery and important research, I felt drawn towards exploring image generation more. 

Here are some papers I've read recently:

### Denoising Diffusion Probabilistic Models

In the landscape of generative artificial intelligence, Denoising Diffusion Probabilistic Models (DDPMs) stand out as a beacon of innovation, merging the worlds of machine learning and thermodynamics in a way that might just make you wonder if a crash course in non-equilibrium thermodynamics is in order. At its core, DDPM operates through a meticulously designed variational bound, which serves as a bridge connecting diffusion probabilistic models with denoising score matching and Langevin dynamics.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/ddpm.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Understanding the Mechanism
DDPMs introduce a progressively lossy decompression scheme, utilizing autoregressive decoding to achieve remarkable results. The essence of diffusion models lies in their structure as parameterized Markov chains, trained through variational inference to master the art of reversing a diffusion process. This technique hinges on learning transitions within the chain, guiding the model to reverse engineer from a state of high entropy back to the original data distribution.

One of the pivotal revelations in the study of diffusion models is their parameterization, revealing an equivalence with denoising score matching across multiple noise levels during the training phase. Although these models deliver exceptional sample quality, they encounter challenges with log likelihood when juxtaposed with likelihood-based models, leading researchers to analyze these dynamics through the prism of lossy compression.

### The Math Behind the Magic
At the heart of diffusion models is a latent variable framework, encapsulated by the equation:

'''insert equation here'''

This equation symbolizes the averaging over all conceivable paths that latent variables could traverse, starting from the original data point and culminating in the learned distribution. The genesis of this journey begins at the "most diffused state," a point where the data's inherent structure has been obliterated.

The Markov chain that defines the reverse process is a series of learned Gaussian transitions, embarking from a state of maximal entropy. The distinct nature of diffusion models stems from their approximation to the posterior, a fixed Markov chain known as the diffusion process, which incrementally introduces Gaussian noise into the data.

### Forward Pass Simplified
Breaking down the forward pass, we see a methodical addition of Gaussian noise at each timestep, transitioning the data from its initial state towards increasing levels of diffusion. This process allows the model to navigate through a multidimensional noise landscape, with each dimension uniformly affected thanks to the identity matrix 

'''insert code here'''

However, the real ingenuity emerges in the model's ability to reparameterize, enabling a tractable closed-form sample through the clever manipulation of variance schedules. Whether adopting a fixed constant or a dynamic schedule, the choice of variance scheduling, from linear to cosine, significantly influences the model's performance, with recent enhancements spotlighting the efficacy of cosine schedules.

### Reverse Diffusion: The Road Back
As the model approaches the end of its diffusion journey, the latent variable resembles an isotropic Gaussian, paving the way for the reverse diffusion process. This process involves approximating the less noisy state from a given noisy image, a task elegantly executed by neural networks trained to predict Gaussian parameters.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/dog-diffusion.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Training and Beyond
Training a DDPM draws parallels with the variational autoencoder (VAE), focusing on optimizing the negative log-likelihood of the training data. This endeavor touches upon concepts like variational lower bounds and Kullback-Leibler divergence, underscoring the intricate dance between achieving tractability and preserving the model's flexibility to capture the rich structure of arbitrary data.

### Leveraging Physics for AI
Interestingly, DDPMs borrow heavily from physics, particularly non-equilibrium statistical physics, to refine their training methodologies. Techniques like Annealed Importance Sampling and Langevin dynamics offer a window into defining Gaussian diffusion processes with target distributions as equilibrium states, showcasing the profound interplay between physics and AI in advancing generative models.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/langevin-dynamics.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Future Directions and Challenges
As we delve deeper into the nuances of DDPMs, questions of efficiency, sample quality, and computational demands come to the fore. The journey from abstract mathematical formulations to practical applications highlights both the potential and the hurdles in harnessing the full power of diffusion models for generative tasks.

### Conclusion
Denoising Diffusion Probabilistic Models represent a fascinating confluence of ideas, from the mathematical intricacies of variational inference to the thermodynamic principles guiding their operation. As we stand on the brink of new discoveries in generative AI, DDPMs offer a promising pathway, challenging us to rethink the boundaries of what's possible in the realm of artificial intelligence.