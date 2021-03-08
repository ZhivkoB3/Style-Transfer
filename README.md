# Trapped in art

***Abstract* - Style transfer has been ground-breaking into the field of computer vision. Neural style transfer is an algorithm that incorporates the style and texture of one image (style image onto another image known as content image) by reducing their overall loss.**

### I. Introduction

Computer vision is a subfield of AI that deals with gaining deeper undestanding of digital images. In computer vision the type of Neural networks that we use are called Convolutional Neural Networks (CNN). CNN consists of layers of small computational units that process visual information hierarchically in a feed-forward manner. Each layer of units can be understood as a collection of image filters each of which extracts a certain feature from the input image. Thus the output of a given layer consists of so-called feature maps: differently filtered version of the input image[1]. The algorithm that is presented in this notebook is A neural Algorithm of Artistic Style in 2015[1] by Leon A. Gatys.

### II. Related work

A. *Style Transfer*

In the original paper the authors use a pretrained VGG19 that was pretrained on ImageNet also an important note is that they have not used the fully connected layers of the network. The layers used to achieve the style transfer are:

*For content reconstruction* - conv1_1(a), conv2_1(b), conv3_1(c), conv4_1(d), conv5_1(e) of the original VGG-Network

This layers have been chosen because in the higher layers of the network detailed pixel information is lost while high level content of the image is preserved

*For style reconstruction* - conv1_1(a), conv1_1 and conv2_1(b), conv1_1, conv2_1 and conv3_1(c), conv1_1 and conv2_1,conv3_1 and conv4_1(d), conv1_1 and conv2_1,conv3_1, conv4_1 and conv5_1(e)

![StyleTransfer](https://miro.medium.com/max/4000/1*l8xUTM0it03UT1rQylo2-A.png)

On top of the CNN representations a new feature space is built that captures the style of an input image. The style representation computes correlations between different features in different layers of the CNN.

Simply put the activations of one of the later layers of the VGG-Network are used for content activations whereas multiple layers throughout the network form the style activations. Using these activaitons the algorithm works towards minimazing the overall loss which is given as:

$J(G) = \alpha$$J_{content}(C, G) + \beta$$J_{style}(S, G)$

where $\alpha$ and $\beta$ are the weighting factors for content and style reconstruction respectively[1] and $J_{content}(C, G)$ is the content loss that measures how the similarity 
between the content of the content image and the generated image and $J_{style}(S, G)$ is the style loss which measures the similarity between the style of the style image and the generated image. Gradient descent minimization is used to minimize $J(G)$ and produce the generated image.

Now, lets have a closer look how everything works.

If we have a hidden layer $l$ of the VGG-network. The activations of this layer for the content image are denoted as $a^{[l](C)}$ and the activations of this layer for the generated image are denoted as $a^{[l](G)}$. The content loss is given as,

$J_{content}(C,G) = \frac{1}{2} ||a^{[l](C)} - a^{[l](G)}||^2 $

If $a^{[l](C)}$ and $a^{[l](G)}$ are similar, both images have similar content and the overall loss will be minimal.

Style loss is slightly more complicated concept. On top of the CNN response in each layer of the network we built a style representation that computes the correlation between different filter responses where the expectation is taken over the spartial extend of the image[1]. In other words if one of the channels correspons to the neuran that has the highest activation when a vertical edge is detected and another channel corresponds to the neuron that has the highest activation when a blue hue is detected. If the channels are correlated the part of the image having the vertical edge will aslo have the blue hue. If the two channels are uncorrelated the part of the image having a vertical edge will not have a blue hue. This gives us a measure of similarity in the style of the style image as compared to the style of the generated image. Therese correlations can be captures with the help of a Gram matrix. The Gram matrix is computed as follows:

$G^{[l](S)}_{kk`} = \sum^{n^{[l]}_{h}}_{i = 1}\sum^{n^{[l]}_{w}}_{j = 1}\alpha^{[l]}_{(i, j, k)}.\alpha^{[l]}_{(i, j, k`)}$

Where $\alpha^{[l]}_{(i, j, k)}$ is the activation at $(i, j, k)$ for hidden layer $l$ where $i$ traverses the height, $j$ traverses the width, and $k$ traverses the channels of the feature map. $G^{[l](S)}$ is a matrix with dimension as $n^{[l]}_c$ x $n^{[l]}_c$, where $G^{[l](S)}$ is the Gram matrix for the style image at hidden layer $l$

It is important to note that we can smoothly regulate the emphasis on either reconstructing the content or the style. A strong emphasis on style will result in images that match the appearence of the artwork, effectively giving a texturised version of it but hardly show any of the photograph`s content. When placing strong emphasis on content one can clearly identify the photograph but the style of the painting is not as well-matched. For a specific pair of source images one can adjust the trade-off between content and style to create visually appealing images.

### III. Conclusion

As presented, coding a Neural Style Transfer neural network in Python is not very complicated (beyond calculating the loss functions). In any case, this is not all, Neural Style Transfer networks do not end here, they offer many more possibilities, from applicating them only to a section of an image using masks and for example changing the background of a photo, applying style to the person in the photo, combining several styles and many more.

References: 

[1] A Neural Algorithm of Artistic Style - https://arxiv.org/abs/1508.06576 <br>
[2] https://keras.io/examples/generative/neural_style_transfer/ <br>
[3] https://github.com/Shashi456/Neural-Style/blob/f226abf09b89fcb1970c167aaa666677cdc8f9f5/Neural%20Style%20Transfer/train_TensorFlow.py#L53
