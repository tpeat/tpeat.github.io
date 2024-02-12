---
layout: page
title: Depth Estimation
description: Simplified depth estimation transformer from monocular lens
img: assets/img/depth.jpg
importance: 2
category: work
related_publications: true
---

---

### Project Objective:
* Design a depth estimation model that uses a single lens
* Collect data for long range depths in a variety of settings
* Evaluate the model for accuracy at depth ranges

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/monodepth.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

I joined this project at Georgia Tech Research Institute (GTRI) after 2 semester from its start. While the model architecture and training pipeline was near complete, the team needed assistance in running experiments and evaluating the models.

### Key Contributions
* Developed a model evaluation script that bins pixelwise errors by depths
* Launched exeriments to evaluate various model configurations on a multi-GPU slurm cluster

---

### Data sourcing

The Karlsruhe Institute of Technology and Toyota Technological Institute (KITTI) dataset is one of the most famous datasets for a variety of vision and autonomous vehicle tasks. This dataset was used as a benchmark for the model on short range data, like that in a city landscape. This GIF shown above is an example of RGB and depth frames from a drive in an urban environment.

To collect long range depth data, Microsoft's AirSim simulator to generate depth data using cameras mounted on drones flying through a variety of scenes at a variety of altitudes. Clouds and wind were disabled from data collection. A combination of rural and city scenes were used for training.

<div class="row">
    <div class="row-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/airsim.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="row-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/airsim2.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Samples of depth data from Microsoft's Airsim
</div>

### Model Selection

Previous attempts at monocular lens depth estimation use some type of scaling to ground the depth predicitons to a ground truth value. That is, depth estimation models are good at predicting relative depths but without knowing how far away one element is, they are often several magnitudes away from truth values. To solve this in a self-supervised manner, without the need of passing any ground truth values to the depth net, the model architecture uses two sub networks during training {% cite monodepth2 %}.

DepthNet to predict depths at a given frame and PerceiverNet to predict the pose between two frames. The pose net is only used during training to constrain the depth estimation network. Per pixel reprojection loss is used to fine tune the model.

Depth Estimation with Simplified Transformers (DEST), uses stacked tranformer endoder blocks with residual connections to a pose net and to the decoder resampling blocks. {% cite yang2022depth %}. Overlapped patch embedding is used to preserve local image context. Inspired by the feature pyramid network, DEST uses a progressively upsampling decoder using bilinear interpolation.

<div class="row">
    <div class="row-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/dest.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Attention in transformer blocks are inherently quadratic time complexity when attending to all tokens in the image. Therefore, the DEST model focuses on simplifying the transformer design to achieve efficient inference. The attention mechanism applies a sequence reduction process to reduce K by reduction ratio $$R^2 $$. It also reaplced softmax with row-wise max pooling. Layer norms were also eliminated in favor of batch normalization which is precomputed from train statistics during inference.

<div class="row">
    <div class="row-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/dest-attention.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

### Model Improvements

Inception style modules were used in replace of MixFFN and as a additional spatial reduction mechanism. Inception modules were used for increased parrellism and feature extraction at different granularities, {% cite szegedy2014going %}.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/inception.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The inception block uses point wise convolutions to reduce the channelewise dimension and then max pooling and strided 3x3 convolutions to reduce the spatial reductions.

### Results

Less than 5% MAE for up to 500m, less than 10% MAE for up to 1000m. This seriously expands the capability from other models trained on the KITTI dataset which rarely goes beyond 80m. 

