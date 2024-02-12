---
layout: page
title: Efficient Object Tracking
description: Tracking airborne objects with Associating Objects with Transformers model
img: assets/img/tracking.jpg
importance: 1
category: work
related_publications: true
---

## Project Objective:
* Track single a airborne object from closing distances
* Leverage transfer learning and finetune on a custom airborne object dataset
* Model must handle states of occlusion/deformity/clutter/multi-object
* Explore state of the art tracking techniques
* Experiment with reinforcement learning based trackers
* Model must fit into a single GPU, use <12G of RAM, and operate at >50FPS for inference time

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/track-tmp.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


### Key contributions:
* Found Amazon Airborn Object Tracking dataset
* Implemented custom dataloader, downloading images from S3, cropping, and segmenting the images into Annotation binary masks
* Conducted full literature review, ultimately choosing Associating Objects with Transformers model architecture as launching point
* Identified key areas of improvement for the model such as the need for a better backbone, reducing history of flights, and using a sparser segmentation head
* Developed hierarchical DINO based encoder backbone
* Leverage TokenMerging to improve efficiency of attention mechanisms by reducing token number without loss of information
* Introduced FlashAttention for enhanced GPU usage
* Launched hundreds of experiments on multi GPU, slurm cluster

More details below on the progress from literature review, dataset choice, augmentations, manipulations, model choice, training architecture, next steps.

### Literature Review

The initial literature review was focused on learning from state of the art tracking techniques as well as identifying the feasibility of using RL for tracking. RL in tracking is not well studied and typicaly deep learning methods are preferred for their quicker inference speeds. 

Key takeaways:
* One stage networks are preferred for end to end training and faster inference time
* IoU and distance from centroid are preferred metrics
* Segmentation is preffered for objects of varying shape, but is slower
* Many models focus on multi-object tracking (MOT) and also perform worse on single-object tracking (SOT)
* LaSOT is most common SOT dataset, VOT2018, GOT-10k honorable mention
* Template matching is extremely fast, Siamese networks use this week to balance speed and accuracy


**Complex Environments and Object Variability:** Effective tracking in distracting environments necessitates handling objects with large variance in shape and scale, and coping with both partial and full occlusions.

**Model Exploration:** 

Template matching {% cite hu2022siammask %}

Why we didn't use RL:

What makes DINO such a good encoder?

How should we handle memory of past states/trajectory?

**Self-Supervised Learning and Transformers:** The exploration of self-supervised learning models like DINO, and the integration of Transformers, suggests a shift towards leveraging these advanced architectures for improved tracking performance, especially in understanding long-range dependencies and spatial-temporal relationships.


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
    Examples from the Segment Anything model.
</div>

The Segmant Anything model is extremely performant on most images but fails on some with extremly small targets, typically outputting the entire bbox as a annotation mask or outputting no annotation {% cite kirillov2023segment %}.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/helicopter.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/copter-seg.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    SAM model applied to sample from Airborne Object Tracking dataset.
</div>

## Model Architecture

The chosen architecture was based on the [Associating Objects with Transformers](https://arxiv.org/pdf/2106.02638.pdf) model.

This model was further improved by the [Decoupling Features in Hierarchical Propagation
for Video Object Segmentation](https://arxiv.org/pdf/2210.09782.pdf) (DeAOT) model. The key difference between the two being a Gated Propogation Module (GPM) that seperates object specific from object agnostic features. Within the GPM, attention is performed on local tokens and on a global tokens which uses historical information {% cite yang2022decoupling %}. The AOT model uses heiarchical propogation to trasfer information from past frames to current frame {% cite yang2021associating %}. The DeAOT model achieves new SOTA on YouTube-VOS, DAVIS 2017, DAVIS 2016, and VOT 2020. On YouTube-VOS, DeAOT achieves 82% accuracy at 52FPS, meeting the project requirements.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/aot.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/deaot.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Visualization of DeAOT architecture
</div>

The creators of the AOT model provide open source code. One benefit of their code base is each model is defined from in a highly cofigurable manner, making it extremly easy to change backbone, latent dimension, number of attention heads, type of attention, etc.

```python
class DefaultModelConfig():
    def __init__(self):
        self.MODEL_NAME = 'AOTDefault'
        self.MODEL_VOS = 'aot'
        self.MODEL_ENGINE = 'aotengine'
        self.MODEL_ALIGN_CORNERS = True
        self.MODEL_ENCODER = 'mobilenetv2'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/mobilenet_v2-b0353104.pth'
        self.MODEL_ENCODER_DIM = [24, 32, 96, 1280]  # 4x, 8x, 16x, 16x
        self.MODEL_ENCODER_EMBEDDING_DIM = 256
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True
        self.MODEL_FREEZE_BN = True
        self.MODEL_FREEZE_BACKBONE = False
        self.MODEL_MAX_OBJ_NUM = 10
        self.MODEL_SELF_HEADS = 8
        self.MODEL_ATT_HEADS = 8
        self.MODEL_LSTT_NUM = 1
        self.MODEL_EPSILON = 1e-5
        self.MODEL_USE_PREV_PROB = False
```

## Model Improvements

One major limitation of the AOT code is that during model evaluation, the `Evaluator` object saves copies of the predicted annotation masks to disk. As we scale up the number of experiments, memory limitations become an issue. Instead of saving the masks, I refactored the `Evaluator` methods to calculated `IoU` and `FPS` on the fly.

In progress:
* Add DINO backbone
* FlashAttention for more efficient GPU usage
* TokenMerging
* Add Layerwise learning rate decay for deepest models
* Sparser segmentation head