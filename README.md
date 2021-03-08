### Style Transfer

Computer vision is a subfield of AI that deals with gaining deeper undestanding of digital images. In computer vision the type of Neural networks that we use are called Convolutional Neural Networks (CNN). CNN consists of layers of small computational units that process visual information hierarchically in a feed-forward manner. Each layer of units can be understood as a collection of image filters each of which extracts a certain feature from the input image. Thus the output of a given layer consists of so-called feature maps: differently filtered version of the input image[1]. The algorithm that is presented here is "A neural Algorithm of Artistic Style" in 2015[1] by Leon A. Gatys.

### Overview of the model

<img src = "https://miro.medium.com/max/1430/1*JAMQmAJ-oPH35D5K4tJvJQ.png">

### Example generated with this code:

<img src = "https://sites.google.com/site/lilyarteia123/data-charts/vincent-van-gogh/image.jpg?attredirects=0"> + <img src = "https://media.overstockart.com/optimized/cache/data/product_images/VG485-1000x1000.jpg"> = <img src = "https://github.com/ZhivkoB3/Trapped-in-art/blob/main/StyleTransfered.png">


### Acknowledgments: 

[1] A Neural Algorithm of Artistic Style - https://arxiv.org/abs/1508.06576 <br>
[2] https://keras.io/examples/generative/neural_style_transfer/ <br>
[3] https://github.com/Shashi456/Neural-Style/blob/f226abf09b89fcb1970c167aaa666677cdc8f9f5/Neural%20Style%20Transfer/train_TensorFlow.py#L53
Style Image: The Scream by Edvard Munch, The Scream
Style Image: Starry Night by Vincent Van Gogh, The Starry Night
