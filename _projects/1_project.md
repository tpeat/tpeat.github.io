---
layout: page
title: Efficient Object Tracking
description: Tracking airborne objects with Associating Objects with Transformers model
img: assets/img/tracking.jpg
importance: 1
category: work
related_publications: true
---

The project description is an object tracking algorithm on airborne data.

### Literature Review

Faster R-CNN:
* Region proposal network
* Fully convolutional
* Two stage detector
* Anchor based approach
* IoU lost function

YOLO:
* Single state detector
* Grid based detector
* End to end training
* Real time processing

### Key Findings:

**Complex Environments and Object Variability:** Effective tracking in distracting environments necessitates handling objects with large variance in shape and scale, and coping with both partial and full occlusions.

**Model Exploration:** The discussion around the DINO and Perceiver models highlights a curiosity for models that adeptly handle complex tracking scenarios. The inquiry into Reinforcement Learning (RL) underscores a desire for models that are not just reactive but can anticipate and adapt to changes in the environment.

**Actor-Critic Reinforcement Learning Architecture:** This approach emerges as a promising solution, combining the strengths of both policy-based and value-based RL. It optimizes tracking by using an actor to explore and a critic to evaluate the actions, guiding the system towards optimal decision-making.

**SiamMask and Faster R-CNN:** These models are highlighted for their effectiveness in real-time object tracking and detection. SiamMask, for instance, offers a fully convolutional Siamese approach to produce bounding boxes and perform segmentation at impressive speeds, while Faster R-CNN innovates with its Region Proposal Network (RPN) for efficient and accurate object detection.

**Deep Q-Learning and Neural Maps for RL:** Incorporating deep Q-networks and structured memory into RL presents a pathway for enhancing object tracking. These approaches refine the decision-making process, allowing for more precise and adaptive tracking strategies.

**Self-Supervised Learning and Transformers:** The exploration of self-supervised learning models like DINO, and the integration of Transformers, suggests a shift towards leveraging these advanced architectures for improved tracking performance, especially in understanding long-range dependencies and spatial-temporal relationships.

**Application to Aerial Tracking:** The unique challenges posed by tracking aerial vehicles, such as variable appearances from different angles and occlusions, necessitate innovative solutions. This includes adaptive template updating and employing lightweight deep vision reinforcement learning for dynamic object tracking from UAV perspectives.

**Datasets and Evaluation:** The review identifies the need for specialized datasets catering to aerial vehicles and suggests methods for creating comprehensive datasets by merging existing ones. Evaluation metrics such as mean overlap precision, tracking speed, and robustness evaluations (SRE and TRE) are crucial for assessing algorithm performance.


### Data sourcing: Amazon Airborne Object Tracking Dataset

The Airborne object tracking dataset consists of 164 hours of flight data, 4943 flight sequences of aroudn 120 seconds each. This equates to over 3.3M images with annotations of 2d bounding box, object class, and distance.

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/obj-track-challenge.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/tracking2.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Challenge logo and samples from dataset
</div>

### Converting bounding box to segmentation mask

Enter Segmant Anything model.

By passing the image along with its respective bounding box to the "Segment Anything" model, it's possible to obtain precise segmentation masks that outline the exact contours of objects within an image. This model leverages advanced deep learning techniques to understand and delineate the object's shape, going beyond the limitations of bounding boxes to provide a pixel-perfect representation. This method not only improves the accuracy of object tracking, especially in complex scenes with overlapping objects, but also facilitates a range of applications that require detailed object shapes, such as advanced image editing, augmented reality, and more sophisticated scene understanding tasks.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/segment.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Examples from the Segment Anything model
</div>

You can also put regular text between your rows of images, even citations {% cite einstein1950meaning %}.
Say you wanted to write a bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, _bled_ for your project, and then... you reveal its glory in the next row of images.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/helicopter.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>

## Architecture

The chosen model architecture was based on the [Associating Objects with Transformers](https://arxiv.org/pdf/2106.02638.pdf) paper.

Which was further improved by the [Decoupling Features in Hierarchical Propagation
for Video Object Segmentation](https://arxiv.org/pdf/2210.09782.pdf) paper.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/aot.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/deaot.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>

## Model Improvements

Completed:
* Pair AOT dataloader with custom dataset
* Launch experiments on slurm cluster
* Disable saving segmasks during evaluation

In progress:
* Add DINO backbone
* FlashAttention for more efficient GPU usage
* TokenMerging
* Add Layerwise learning rate decay for deepest models
* Sparser segmentation head