import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# define the standalone discriminator model
def define_discriminator(in_shape=(128,128,3)):
    model = keras.models.Sequential()
    # normal
    model.add(keras.layers.Conv2D(64, (3,3), padding='same', input_shape=in_shape))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # compile model
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    # 损失函数           二元分类器                    优化器             准确率
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model

def read_image():
    images = []
    for dir_item in os.listdir('/home/litbo/PycharmProjects/view/head/two/ktFaces'):
        full_path = os.path.abspath(os.path.join('/home/litbo/PycharmProjects/view/head/two/ktFaces',dir_item))
        image = cv2.imread(full_path)
        # image = cv2.resize(image,(32,32))
        images.append(image)
    return images
# load and prepare cifar10 training images
def load_real_samples():
    # load cifar10 dataset
    trainX = read_image()
    trainX =np.array(trainX)
    # convert from unsigned ints to floats
    X = trainX.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y


def generate_fake_samples1(n_samples):
    # generate uniform random numbers in [0,1]
    X = np.random.rand(128* 128 * 3 * n_samples)

    # update to have the range [-1, 1]
    X = -1 + X * 2
    # reshape into a batch of color images
    X = X.reshape((n_samples, 128, 128, 3))
    # generate 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, y


# train the discriminator model
def train_discriminator(model, dataset, n_iter=20, n_batch=128):
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_iter):
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator on real samples
        _, real_acc = model.train_on_batch(X_real, y_real)
        # generate 'fake' examples
        X_fake, y_fake = generate_fake_samples1(half_batch)
        # update discriminator on fake samples
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
        # summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

def test_train_discriminator():
    # define the discriminator model
    model = define_discriminator()
    # load image data
    dataset = load_real_samples()
    # fit the model
    train_discriminator(model, dataset)


# define the standalone generator model
def define_generator(latent_dim):
    model = keras.models.Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(keras.layers.Dense(n_nodes, input_dim=latent_dim))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # upsample to 64x64
    model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # upsample to 128x128
    model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # output layer
    model.add(keras.layers.Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    # 噪声z所在的空间
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, y


def show_fake_sample():
    # size of the latent space
    latent_dim = 100
    # define the discriminator model
    model = define_generator(latent_dim)
    # generate samples
    n_samples = 49
    X, _ = generate_fake_samples(model, latent_dim, n_samples)
    # scale pixel values from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot the generated samples
    for i in range(n_samples):
        # define subplot
        plt.subplot(7, 7, 1 + i)
        # turn off axis labels
        plt.axis('off')
        # plot single image
        plt.imshow(X[i])
    # show the figure
    plt.show()


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # 判别器的模型参数不被更新
    d_model.trainable = False
    # connect them
    model = tf.keras.models.Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    # beta_1：float，0 <beta <1。一般接近1。一阶矩估计的指数衰减率
    # beta_2：float，0 <beta <1。一般接近1。二阶矩估计的指数衰减率
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def show_gan_module():
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    # summarize gan model
    gan_model.summary()



# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    #save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 200 +1)
    g_model.save(filename)


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=40, n_batch=128):
    gan_model.load_weights('generator_model_200.h5',by_name=True)
    # json_string = gan_model.to_json()
    # gan_model = json_string(json_string)
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # 更新判别器模型权重
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # 生成器生成假数据
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # 更新判别器模型权重
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


def train_gan():
    # size of the latent space
    latent_dim = 100
    # 创建判别器
    d_model = define_discriminator()
    # 创建生成器
    g_model = define_generator(latent_dim)
    # 创建对抗网络
    gan_model = define_gan(g_model, d_model)
    # 加载数据集
    dataset = load_real_samples()
    # 训练模型
    train(g_model, d_model, gan_model, dataset, latent_dim)



# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# plot the generated images
def create_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :])
    plt.show()

def show_imgs_for_final_generator_model():
    # load model
    model = tf.keras.models.load_model('generator_model_200.h5')
    # generate images
    latent_points = generate_latent_points(100, 100)
    # generate images
    X = model.predict(latent_points)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot the result
    create_plot(X, 10)

def show_single_imgs():
    model = tf.keras.models.load_model('generator_model_180.h5')
    # all 0s
    vector = np.asarray([[0.75 for _ in range(100)]])
    # generate image
    X = model.predict(vector)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot the result
    plt.imshow(X[0, :, :])
    plt.show()

if __name__ == '__main__':
    # show_imgs_for_final_generator_model()
    # define_discriminator()
    # test_train_discriminator()
    # show_fake_sample()
    # show_gan_module()
    train_gan()
    # show_imgs_for_final_generator_model()
    # show_single_imgs()
    # res = read_image()
    # print(res[0].shape)
