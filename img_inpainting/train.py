import os
import datetime
import click
import numpy as np
import cv2

from utils import load_data
from model import generator_model, discriminator_model, build_vgg, generator_containing_discriminator_multiple_outputs
from keras.optimizers import Adam

BASE_DIR = '/home/alyssa/PythonProjects/occluded/key_code/img_inpainting/weights/'


def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


def train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates=5):
    g = generator_model()
    d = discriminator_model()
    vgg = build_vgg()
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d, vgg)

    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    optimizer = Adam(1E-4, 0.5)
    vgg.trainable = False
    vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    d.trainable = True
    d.compile(optimizer=d_opt, loss='binary_crossentropy')
    d.trainable = False
    loss = ['mae', 'mse', 'binary_crossentropy']
    loss_weights = [0.1, 100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.zeros((batch_size, 1))

    for epoch in range(epoch_num):
        print('epoch: {}/{}'.format(epoch, epoch_num))

        y_pre, x_pre, mask = load_data(batch_size)

        d_losses = []
        d_on_g_losses = []

        generated_images = g.predict(x=x_pre, batch_size=batch_size)

        for _ in range(critic_updates):
            d_loss_real = d.train_on_batch(y_pre, output_true_batch)
            d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            d_losses.append(d_loss)
        print('batch {} d_loss : {}'.format(epoch, np.mean(d_losses)))

        d.trainable = False

        real_result = mask * y_pre
        y_features = vgg.predict(y_pre)

        d_on_g_loss = d_on_g.train_on_batch([x_pre, mask], [real_result, y_features, output_true_batch])
        d_on_g_losses.append(d_on_g_loss)
        print('batch {} d_on_g_loss : {}'.format(epoch, d_on_g_loss))

        d.trainable = True

        if epoch % 100 == 0:
            generated = np.array([(img+1)*127.5 for img in generated_images])
            full = np.array([(img+1)*127.5 for img in y_pre])
            blur = np.array([(img+1)*127.5 for img in x_pre])

            for i in range(3):
                img_ge = generated[i, :, :, :]
                img_fu = full[i, :, :, :]
                img_bl = blur[i, :, :, :]
                output = np.concatenate((img_ge, img_fu, img_bl), axis=1)
                cv2.imwrite('/home/alyssa/PythonProjects/occluded/key_code/img_inpainting/out/'+str(epoch)+'_'+str(i)+'.jpg', output)

        if(epoch > 10000 and epoch % 1000 == 0):
            save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))


@click.command()
@click.option('--n_images', default=-1, help='Number of images to load for training')
@click.option('--batch_size', default=128, help='Size of batch')
@click.option('--epoch_num', default=60001, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates)


if __name__ == '__main__':
    train_command()
