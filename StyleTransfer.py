class StyleTransfer():

    def __init__(self, number_of_epochs=1000, content_weight=1e3, style_weight=1e-2, enable_gpu=False, verbose=False):
        self.number_of_epochs = number_of_epochs
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_layers = ["block5_conv2"]
        self.style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
        self.verbose = verbose
        self.enable_gpu = enable_gpu
        self._set_device()

    def perform(self, content_file, style_file):
        return self._perform(content_file, style_file)

    def display_this(self, image):
        return self._display_this(image)
    
    def _set_device(self):
        """
        Sets device as CPU or GPU
        """
        if(self.enable_gpu):
            device_name = tf.test.gpu_device_name()
            if device_name != '/device:GPU:0':
              raise SystemError('GPU device not found')
            if(self.verbose):
                print('Found GPU at: {}'.format(device_name))
            self.device_name = "/gpu:0"
        else:
            self.device_name = "/cpu:0"

    def _load_image(self, filename):
        img = Image.open(filename)
        img = img.resize((img.size[0], img.size[1]), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32)
        if(img.shape[2] == 4):
            img = img[:, :, :3]
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def _deprocess_image(self, processed_image):
        x = processed_image.copy()
        if(len(x.shape) == 4):
            x = np.squeeze(x)
        assert(len(x.shape) == 3)

        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.680

        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def _get_model(self):

        vgg_model = tf.keras.applications.vgg19.VGG19(include_top = False, weights='imagenet')
        vgg_model.trainable = False
        if(self.verbose == True):
            print(vgg_model.summary())

        content_outputs = [vgg_model.get_layer(layer).output for layer in self.content_layers]
        style_outputs = [vgg_model.get_layer(layer).output for layer in self.style_layers]

        model_outputs = style_outputs + content_outputs
        return models.Model(vgg_model.inputs, model_outputs)

    def _get_activations(self, model, content_file, style_file):

        content_image = self._load_image(content_file)
        style_image = self._load_image(style_file)

        content_outputs = model(content_image)
        style_outputs = model(style_image)

        content_image_activations = [content_layer[0] for content_layer in content_outputs[len(self.style_layers):]]
        style_image_activations = [style_layer[0] for style_layer in style_outputs[:len(self.style_layers)]]

        return content_image_activations, style_image_activations

    def _content_loss_computation(self, content_image_activations, generated_image_activations):
        return tf.reduce_mean(tf.square(content_image_activations - generated_image_activations))

    def _gram_matrix_computation(self, input_activations):
        channels = input_activations.shape[-1]
        activations = tf.reshape(input_activations, [-1, channels])
        number_of_activations = activations.shape[0]
        gram_matrix = tf.matmul(activations, activations, transpose_a=True)
        gram_matrix = gram_matrix / tf.cast(number_of_activations, tf.float32)
        return gram_matrix

    def _style_loss_computation(self, gram_matrix_style_image, gram_matrix_generated_image):
        height, width = gram_matrix_style_image.get_shape().as_list()
        return tf.reduce_mean(tf.square(gram_matrix_style_image - gram_matrix_generated_image))

    def _compute_overall_loss(self, model, loss_weights, generated_image, content_image_activations, style_image_activations):

        content_weight, style_weight = loss_weights

        model_outputs = model(generated_image)

        generated_image_activations_content = [content_layer[0] for content_layer in model_outputs[len(self.style_layers):]]
        generated_image_activations_style = [style_layer[0] for style_layer in model_outputs[:len(self.style_layers)]]

        gram_matrices_style_image = [self._gram_matrix_computation(activation) for activation in style_image_activations]
        gram_matrices_generated_image = [self._gram_matrix_computation(activation) for activation in generated_image_activations_style]

        style_loss = 0
        content_loss = 0

        weight_per_content_layer = 1.0 / float(len(self.content_layers))
        for a, b in zip(content_image_activations, generated_image_activations_content):
            content_loss = content_loss + (weight_per_content_layer * self._content_loss_computation(a, b))

        weight_per_style_layer = 1.0 / float(len(self.style_layers))
        for a, b in zip(gram_matrices_style_image, gram_matrices_generated_image):
            style_loss = style_loss + (weight_per_style_layer * self._style_loss_computation(a, b))

        content_loss = content_loss * content_weight
        style_loss = style_loss * style_weight
        overall_loss = content_loss + style_loss

        return overall_loss, content_loss, style_loss

    def _compute_gradients(self, parameters):
        with tf.GradientTape() as g:
            losses = self._compute_overall_loss(**parameters)
        overall_loss = losses[0]
        return g.gradient(overall_loss, parameters['generated_image']), losses

    def _perform(self, content_file, style_file):
        with tf.device(self.device_name):
          model = self._get_model()
          for layer in model.layers:
              layer.trainable = False
          content_image_activations, style_image_activations = self._get_activations(model, content_file, style_file)
          content_image = self._load_image(content_file)
          generated_image = self._load_image(content_file)
          generated_image = tf.Variable(generated_image, dtype=tf.float32)
          optimizer = tf.optimizers.Adam(learning_rate = 2.5)
          loss_weights = (self.content_weight, self.style_weight)
          final_loss, final_image = np.inf, None
          parameters = {
          "model" : model,
          "loss_weights": loss_weights, 
          "generated_image": generated_image,
          "content_image_activations": content_image_activations,
          "style_image_activations": style_image_activations
          }
          norm_means = np.array([103.939, 116.779, 123.680])
          min_value = -norm_means
          max_value = 255 - norm_means
          for epoch in range(self.number_of_epochs):
              gradients, losses = self._compute_gradients(parameters)
              overall_loss, content_loss, style_loss = losses
              optimizer.apply_gradients([(gradients, generated_image)])
              clipped_image = tf.clip_by_value(generated_image, min_value, max_value)
              generated_image.assign(clipped_image)

              if overall_loss < final_loss:
                final_loss = overall_loss
                final_image = generated_image.numpy()
                final_image = self._deprocess_image(final_image)
              if(self.verbose == True):
                  print("Epoch Number %d : Completed" % (epoch+1))
        return final_image

    def _display_this(self, image):
        figure(figsize = (10, 8))
        plt.imshow(image), plt.title('Style transfered'), plt.axis('off')
        plt.show()
        return
