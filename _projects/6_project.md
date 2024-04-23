---
layout: page
title: Messi-Anything
description:
img: assets/img/messi.jpeg
importance: 1
category: fun
related_publications: true
---

## Project Goal

Create an end to end model that take an RGB image, segments out humans if they exist, and inpaints the space with a generated image of Messi.

The original name was Segment-And-Replace, but we thought something a little more fun, hence Messi everything.

How did this idea come about?

- I have experience with segmentation and deep vision transformers
- I want to learn more about image generation models
- I play soccer and my favorite player is Messi

---

## Combining the latent space:

My idea was simple, we spend so much time encoding the input images into a latent space, we can leverage work already performed by the segmentation model when creating the masks to aide the gen model.

I figured because the Hiera encoder reaches a smaller spatial dimension quicker than the generative UNet, with dimensions (256x256 --> 64x64 --> 32x32 --> 16x16 --> 8x8) that we could perform cross attention between the lower dimension space without too much overhead

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/model-arch.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

You may have already noted that the dimension don't match even closely: trying to perform cross attention between (32x32x192) and (128x128x512) tensors is a tricky task. Projecting the tensors channels into a similar dimension is a well-known tactic, but what to do about the spatial dimension? I could interpolate the smallest to match the largest, or downsample the largest to match the smallest, some combination of both?

---

## Data

For the segmentation model, I wanted to use a well established and easily available dataset. I chose MSCOCO because it contains a decent amount of 'people' labels. I wrote this quick script to download the instance json, filter by people, and then download associated masks and labels.

```python
max_people = 3  # avoid overcrowing of people in the images

coco = COCO(annFile)

# # get category ids for everythign containing person
catIds = coco.getCatIds(catNms=['person'])
# get images for person category
imgIds = coco.getImgIds(catIds=catIds)

for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    annotations = coco.loadAnns(annIds)
    # this avoids crowded frames
    if len(annotations) > max_people:
        continue

    response = requests.get(img['coco_url'])
    image = Image.open(BytesIO(response.content))
    image.save(f'{dataDir}/{dataType}/{img["file_name"]}')

    if annotations:
        mask = np.zeros((img['height'], img['width']))
        for ann in annotations:
            mask += coco.annToMask(ann)
        # save mask ...
```

<p></p>

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mscoco-ex.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Example images from MS coco, the furthest right would not pass the max_people threshold
</div>

For the Messi dataset, the team found a few Messi datasets online from kaggle and parsed them together. The final count was 1087 images. The resolutions and dimensions of the images were diverse and wildly different.

Note that there are no labels associated with the data required for the generative model, making the generative part of our model fully unsupervised. This is because the model is merely learning the distribution of the input images, so that it can map a noise vector to a sample from the distribution that is novel and realistic.

---

## Closer Look at the Segmentation Model

The encoder is based on the Hiera model introduced by Dan Bolya and Meta. The model is about 53M paramters and I loaded pretrained weights adapted from DINOv2 (also product of Meta).

The decoder is an extremly simple Feature Pyramid Network (FPN) that has 3 blocks of upsampling and intepolation from

```
8x8[input] -> 16x16 -> 32x32 -> 64x64
```

Then, I added transformer blocks (from the timm library) in between the encoder and decoder to create a more powerful representation in the latent space.

The combined model looks like so:

```python
class Model(torch.nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, depth=10):
        super().__init__()
        self.encoder = create_hiera_model()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.blocks = torch.nn.Sequential(*[
            Block(embed_dim, num_heads) for _ in range(depth)
        ])
        self.decoder = FPNSegmentationHead(embed_dim, 1, decode_intermediate_input=False, shortcut_dims=[96,192,384,768])

    def forward(self, x):
        intermediates = self.encoder(x, return_intermediates=True)
        shortcuts = []
        x = intermediates[-1]
        for i in intermediates:
            shortcuts.append(i.permute(0, 3, 1, 2))
        x = self.blocks(x).permute(0, 3, 1, 2)
        x = self.decoder([x], shortcuts)
        return x
```

### Training Details

With access to the Georgia Tech PACE-Ice super compute cluster, we were able to train the segmentation model for 200 epochs on a single H100 with 80GB RAM.

The initial learning = 0.001 and optimizer was Adam.

A variety of loss function were tested for best results but ultimately only BCELossWithLogits was used.

Other variations include IoULoss using a SoftJaccardIndex and DiceLoss

Note: that IOU = Intersection over union which can be done in pytorch simply using:

```python
intersection = (pred * target).sum(1,2)
(union = pred + target).sum(1,2)
mean_iou = (intersection / union).mean()
```

Naturally you will want to add in small constants for numeric stability and be cautious of the dimension of your tensors.

Iou ranges between 0-1 with 1 being perfect.

To turn Iou into a loss function, you need only subtract iou from 1

Below you will find training curves for 100 epochs.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/bce_loss.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/iou-per-epoch.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    BCE Loss on the left with IOU for eval in the right
</div>

### Evaluation Details

A test set of 100 images was withheld from training. Below is the performance on a image from the test set.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/test-messi.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Example output on test image
</div>

You may notice the choppy edges on the edge of the mask. This is the result of the FPN only have 3 upsampling convolution + interpolation blocks, so the final output from the FPN is only 64x64, then the rest is interpolated (makes blocky represenation).

The final convolution back to the original dimension was removed for inference speed.

---

## Closer look at Consistency Models

The idea of consistency models originates from Dr. Yang Song, from OpenAI.

NOTE: mid project, I switched from A100 GPUs to H100 GPUs because there were 8 times at many H100s as A100s. This meant I had to find a mapping between the modules such as cuda available on A100 to the one's available on H100. Furthermore, I had to recreate my environment with these new modules and thus I upgraded from `pytorch=2.0.1` to `pytorch=2.1.0`. This meant that the version of `flash-attn` in the `setup.py` file for the consistency-models was out of date and the FlashAttention mechanisms in the UNet would no longer work. The speedup offered by hardware efficient attention is signfigant to I wrote a fix:

```python
class TristanAttention(nn.Module):
    """
    Implementation of F.scaled_... for flasha ttention
    using pytorch > 2.0, they have legacy version outdated for current cuda version

    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        from einops import rearrange

        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        self.attention_dropout = attention_dropout
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads
        )
        query, key, value = qkv.chunk(3, 2)
        query, key, value = query.squeeze(2), key.squeeze(2), value.squeeze(2)
        # print(query.shape)
        with th.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
            ) and th.cuda.amp.autocast():
            # th.cuda.synchronize() # questionable call to sync, Don't want to force all models to schedule this
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=self.attention_dropout, attn_mask=attn_mask, is_causal=self.causal)
            # print("out shape:", out.shape)
        return self.rearrange(out, "b s h d -> b (h d) s")
```

Then I overrode the Attention module select when `use_flash=True` in the Unet.

As detailed in the diagram above, the UNet architecture relies on a combination of ResNet and Attention blocks in sequence.

Attention blocks are only used when in the `attention_resolution` list, which is typically 32 or below. This is done so that we aren't wasting compute on computing attention (which is already an $O(n^2)$ operation) on tensors with large spatial dimensions.

Therefore, most of the model is of the form: ResNetBlock -> ResNetBlock -> ResNetBlockWithDownsampling, or vice versa when in the decoder part of the network.

### Training details

As mentioned earlier, the dataset consists of 1087 images of Messi.

I created two models of different resolutions but `CUDA OOM` errors forced me to make two other tweaks:

Image A:

```
image_size = 256
model_dimension = 256
batch_size = 16
```

Image B:

```
image_size = 512
model_dimension = 192
batch_size = 4
```

I trained both models from scratch using a exponential moving average (EMA) with a teacher model, student model, and 3 versions of EMA models tracking models over a number of historical steps.

I trained on a single H100 80GB GPU with the following environment variables:

```pytyhon
python = 3.10
gcc = 12.3
openmpi = 4.1.5
cuda = 12.1.1
pytorch = 2.1.0
```

The training lasted for 16 hours and in that time model A reached 100k steps which accounting for its batch_size means it saw 1.6M images and model B reached 150k steps means its saw 600k images.

---

## Consistency Model Results

The typical evaluation metric for quality of generative image models is Frechet Inception Distance (FID). FID was calculated using the `pytorch-fid` library.

<p></p>

<div class="row justify-content-sm-center">
    <table
    data-toggle="table"
    data-url="{{ '/assets/json/table_data.json' | relative_url }}">
    <thead>
        <tr>
        <th data-field="steps">Steps</th>
        <th data-field="fid">FID</th>
        </tr>
    </thead>
    </table>
</div>

<p></p>

Note: the `pytorch-fid` library requires that you use at least `2048` images for evaluation, both in original dataset and generated dataset. However, I only used 1087 for training. Therefore, the naive fix was to create copies of the images until I met the target count. However, this meant that I should also only sample 1087 from the generative model and then also create copies??? Idk

<p></p>

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/best_ex_1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/best_ex_2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/best_ex_3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/best_ex_4.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/best_ex_5.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/best_ex_6.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/best_ex_7.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/best_ex_8.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/best_ex_10.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Visual inspection shows that its got a lot of things right, and obvious things to improve upon. That is, you can tell its messi. It's got the Argentina and Barcelona kits down,mabye it's the pretty similar stripe pattern. You can see glimpses of the new pink Inter Miami kit in many of the sample too.

Ears, hands and different poses will always be difficult for generation models and I think mine needs a few hundred thousand more epochs before it starts to get this down.

For this small selection of "best" images, I went through the original dataset and find examples close to them and couldn't find much, which is a good sign that its truly syntehsizing across the entire distribution rather than just outputing the training data.

---

## Limitations and feedback:

- **Noisy data:** Ah yes, a tale as old as time. A machine learning researcher so focused on his model, that he doesn't invest in his data until its too late... that's not me. Look at some of my favorite examples I found in the training set:

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/not_even_messi.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/not_even_messi_2.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/barely_messi.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- **Limited storgage.** My storage use on the PACE-ICE computer cluster is limited to 300GB, thus when checkpointing every 5k iterations to ensure I don't lose valuable hours if the training fails by making backups often, I run out of storage in 150k iterations or 15 checkpoints (each checkpoint 2GB for 5 state dicts, assuming I start at 50% storage with dataset and other files consuming the system). This turns out to be one the largest detriments to the project because I could not have concurrent runs because it would fill the storage even quicker. Furthermore, all these problems are only for the generative model! We still needed to train a segmentation model.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/pace-storage.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- **Broken MPI modules.** My model would simply hang when I tried to allocate more than one GPU and use MPI to coordinate the distribution of model and input data.
- **Limited access to compute.** It was a constant battle tring to acquire a large enough GPU for my purposes. When loading the teacher and student model, optimizer, and then pulling data, I was getting dangerously close to the 80GB datalimit on the largest GPUs that I could get for free. This raises many questions about how industry handles such large models.
- **512 pixel resolution was just too big.** After 100k iterations, the model was only starting to form the notion of a face in the output.
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/512x512.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
- **Resizing instead of cropping!** Naively, I designed the FID score to resize all of the original images to be 64x64 so that they could be evaluated against the image generated by the model. Here's why this was a bad idea:

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/messi_small_1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/messi_small_2.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/messi_small_3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

As you can see, the dimensions of Messi are all out of wack. However, this issue was corrected by the final.

- **Limitations of center crop!** Performing center crop fixes the dimension warp, but introduces issues where we miss Messi's face and wind up with some random focal point.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/bad_crop_1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/bad_crop_2.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/bad_crop_3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- **Pytorch-FID calculations**: Pytorch-FID intrinsically requires a minimum of 2048 images. Makes sense. You wouldn't want your eval code to be judging similarity between distributions based on samples that are too small. Problem is: our original dataset was less than 2048, it was 1087 images of Messi. Quick fix: I made duplicates of some images in the dataset until I hit that target_count. At first I did a similar thing with the generative images, however starting from only 500 samples. Meaning if those 500 samples contained some real duds, they were being duplicated multiple times but also if I had a really good batch, they were unfairly weighted against the original sample. Sampling with a multi-step approach (40 steps) does take time (~ 1 sample/sec at 256 resolution), so I tried to cheat the system by requiring it to only make 1087 sample, then call the sample duplicate_image function that I used on the original dataset.

## Inpainting

The inpainting script provided in the constiency model repo was insufficient for my purposes, because it randomly creates some rectangle in an image. But I want to use masks from the segmentation model to tell the consistency model where to fill.

My first attempt got suprisingly good results, I could see the human face starting to form.

I soon realized it was uncovering features that were present in the original image, which the consitency model should have no knowledge of, so I caught my bug.

I was created my initial noise vector from the original image, but rather it should be drawn from the generator.

```python
x_T = generator.randn(*shape, device=device) * args.sigma_max
```

Then the inpainting function makes a forward pass through the consistency model and adds the preciction only the part where there is no mask as noted by `x1 * (1 - masks)`

```python
def replacement(x0, x1):
        x_mix = x0 * masks + x1 * (1 - masks)
        return x_mix

images = replacement(images, -th.ones_like(images))

# Convert the time schedule based on the given rho
t_max_rho = t_max ** (1 / rho)
t_min_rho = t_min ** (1 / rho)
s_in = x.new_ones([x.shape[0]])

for i in range(len(ts) - 1):
    t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
    # call to model
    x0 = distiller(x, t * s_in)
    x0 = th.clamp(x0, -1.0, 1.0)
    # inpaint the image with prediction
    x0 = replacement(images, x0)
    next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
    next_t = np.clip(next_t, t_min, t_max)
    x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

return x, images
```

But, I was getting very _colorful_ results!

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/inpaint_v0.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/inpaint_v2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left is original image minus its mask, right is sample ouputs from the inpainter
</div>

Has to be a normalization issue, right?

## Appendix

Space for other funny images

Check out this picasso!

<div class="row justify-content-sm-center">
    <div class="col-lg mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/messi-picasso.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
