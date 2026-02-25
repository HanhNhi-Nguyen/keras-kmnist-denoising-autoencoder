#model.py
import tensorflow as tf

def build_autoencoder():
    inp = tf.keras.Input(shape=(28, 28, 1), name="noisy")

    x = tf.keras.layers.Conv2D(
        filters=8, kernel_size=3, strides=2,
        padding="same", activation="relu", name="enc_conv1"
    )(inp)

    x = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=2,
        padding="same", activation="relu", name="enc_conv2"
    )(x)

    z = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1,
        padding="same", activation="relu", name="bottleneck"
    )(x)

    x = tf.keras.layers.Conv2DTranspose(
        filters=8, kernel_size=3, strides=2,
        padding="same", activation="relu", name="dec_tconv1"
    )(z)

    out = tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=2,
        padding="same", activation="sigmoid", name="dec_tconv2"
    )(x)

    model = tf.keras.Model(inp, out, name="kmnist_dae_tflm_int8")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model
