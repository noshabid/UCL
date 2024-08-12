import tensorflow as tf
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(2000, activation='relu'),
            layers.Dense(500, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(500, activation='relu'),
            layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

'''
class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(2000, activation='relu'),
            layers.Dense(500, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(500, activation='relu'),
            layers.Dense(tf.math.reduce_prod(shape), activation='linear'),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(500, activation='swish'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(500, activation='swish'),
            layers.Dense(tf.math.reduce_prod(shape), activation="sigmoid"),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(2000, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(2000, activation='relu'),
            layers.Dense(tf.math.reduce_prod(shape), activation='linear'),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(500, activation='swish'),
            layers.Dense(latent_dim, activation='relu', activity_regularizer=regularizers.L1(0.0001)),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(500, activation='swish'),
            layers.Dense(tf.math.reduce_prod(shape), activation="sigmoid"),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(4000, activation='relu'),
            layers.Dense(2000, activation='relu'),
            layers.Dense(500, activation='relu'),
            layers.Dense(250, activation='relu'),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(250, activation='relu'),
            layers.Dense(500, activation='relu'),
            layers.Dense(2000, activation='relu'),
            layers.Dense(4000, activation='relu'),
            layers.Dense(tf.math.reduce_prod(shape), activation="linear"),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(2000, activation='swish'),
            layers.Dense(500, activation='swish'),
            layers.Dense(250, activation='swish'),
            layers.Dense(latent_dim, activation='relu', activity_regularizer=regularizers.L1(0.0001)),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(250, activation='swish'),
            layers.Dense(500, activation='swish'),
            layers.Dense(2000, activation='swish'),
            layers.Dense(tf.math.reduce_prod(shape), activation="linear"),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(2000, activation='relu'),
            layers.Dense(500, activation='relu'),
            layers.Dense(250, activation='relu'),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(250, activation='relu'),
            layers.Dense(500, activation='relu'),
            layers.Dense(2000, activation='relu'),
            layers.Dense(tf.math.reduce_prod(shape), activation="linear"),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

'''