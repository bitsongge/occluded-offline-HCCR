from keras.layers import Input, Activation, Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras.applications import VGG19

from layer_utils import ReflectionPadding2D

image_shape = (64, 64, 3)

ngf = 32
ndf = 32
output_nc = 3
input_shape_discriminator = (64, 64, 3)


def generator_model():
    inputs = Input(shape=image_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    n_downsampling = 4
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
        x = Activation('relu')(x)
          
    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(filters=output_nc, kernel_size=(7, 7), padding='valid')(x)
    x = Activation('tanh')(x)

    outputs = x

    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    plot_model(model, to_file='generator.png', show_shapes=True)
    return model


def discriminator_model():
    n_layers, use_sigmoid = 4, False
    inputs = Input(shape=input_shape_discriminator)

    x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(1, n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    plot_model(model, to_file='discriminator.png', show_shapes=True)
    return model


def build_vgg():
    vgg = VGG19(weights="imagenet")
    # Set outputs to outputs of last conv. layer in block 3
    # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    vgg.outputs = [vgg.layers[9].output]

    img = Input(shape=input_shape_discriminator)
    # Extract image features
    img_features = vgg(img)

    return Model(img, img_features)


def generator_containing_discriminator_multiple_outputs(generator, discriminator, vgg):
    input_images = Input(shape=image_shape)
    mask = Input(shape=image_shape)

    generated_images = generator(input_images)
    fake_features = vgg(generated_images)
    fake_result = Multiply()([mask, generated_images])
    outputs = discriminator(generated_images)

    model = Model(inputs=[input_images, mask], outputs=[fake_result, fake_features, outputs])
    return model


if __name__ == '__main__':
    g = generator_model()
    d = discriminator_model()
    # g.summary()
    # d = discriminator_model()
    # g.save('/home/alyssa/PythonProjects/rain/deblur-gan-master/g_model.h5')
    # d.save('/home/alyssa/PythonProjects/rain/deblur-gan-master/d_model.h5')
