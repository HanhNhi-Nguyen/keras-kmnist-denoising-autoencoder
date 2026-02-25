#quant.py
import os
import tensorflow as tf

from .data import make_datasets

def representative_data_gen(ds_calib):
    for x_noisy, _ in ds_calib.take(1000):
        yield [tf.cast(x_noisy, tf.float32)]

def main():
    os.makedirs("export_int8", exist_ok=True)

    # Load trained float model (produced by src.train)
    model = tf.keras.models.load_model("results/float_model.keras")

    _, _, ds_calib = make_datasets()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(ds_calib)

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()

    tflite_path = os.path.join("export_int8", "kmnist_dae_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_int8)

    print("Saved:", tflite_path, "size:", len(tflite_int8), "bytes")

    # Print operator list (for embedded / resolver auditing)
    tf.lite.experimental.Analyzer.analyze(model_path=tflite_path)

if __name__ == "__main__":
    main()
