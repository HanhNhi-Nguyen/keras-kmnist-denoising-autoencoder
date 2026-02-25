#data.py
import tensorflow as tf
import tensorflow_datasets as tfds

IMG_H, IMG_W = 28, 28
BATCH = 128
SEED = 42

def add_noise(x, seed):
    noise = tf.random.stateless_normal(
        tf.shape(x),
        seed=seed,
        mean=0.0,
        stddev=0.35
    )
    return tf.clip_by_value(x + noise, 0.0, 1.0)

def preprocess(example, idx):
    x = tf.cast(example["image"], tf.float32) / 255.0
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
    x_noisy = add_noise(x, seed)
    return x_noisy, x

def make_datasets():
    ds_train_raw = tfds.load("kmnist", split="train", shuffle_files=True)
    ds_test_raw  = tfds.load("kmnist", split="test",  shuffle_files=False)

    ds_train = (ds_train_raw
                .shuffle(60000, seed=SEED)
                .enumerate()
                .map(lambda i, ex: preprocess(ex, i),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH)
                .prefetch(tf.data.AUTOTUNE))

    ds_test = (ds_test_raw
               .enumerate()
               .map(lambda i, ex: preprocess(ex, i),
                    num_parallel_calls=tf.data.AUTOTUNE)
               .batch(BATCH)
               .prefetch(tf.data.AUTOTUNE))

    ds_calib = (tfds.load("kmnist", split="train[:2000]", shuffle_files=False)
                .enumerate()
                .map(lambda i, ex: preprocess(ex, i),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .batch(1))

    return ds_train, ds_test, ds_calib
