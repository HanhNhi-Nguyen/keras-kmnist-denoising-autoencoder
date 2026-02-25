#evaluate.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .data import make_datasets

def main():
    _, ds_test, _ = make_datasets()

    # Load float model for reference
    model = tf.keras.models.load_model("results/float_model.keras")

    tflite_path = "export_int8/kmnist_dae_int8.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    print("Input dtype/shape:", in_det["dtype"], in_det["shape"], "quant:", in_det["quantization"])
    print("Output dtype/shape:", out_det["dtype"], out_det["shape"], "quant:", out_det["quantization"])

    num_eval = 8
    x_noisy, x_clean = next(iter(ds_test.unbatch().batch(num_eval)))

    in_scale, in_zero = in_det["quantization"]
    xq = tf.cast(tf.round(x_noisy / in_scale + in_zero), tf.int8).numpy()

    yq_list = []
    for i in range(num_eval):
        interpreter.set_tensor(in_det["index"], xq[i:i+1])
        interpreter.invoke()
        yq_list.append(interpreter.get_tensor(out_det["index"]))
    yq = np.concatenate(yq_list, axis=0)

    out_scale, out_zero = out_det["quantization"]
    y_hat = (yq.astype(np.float32) - out_zero) * out_scale
    y_hat = np.clip(y_hat, 0.0, 1.0)

    mse = np.mean((y_hat - x_clean.numpy())**2)
    print("INT8 TFLite recon MSE (on", num_eval, "samples):", mse)

    # Debug plots
    num_show = 6
    x_noisy, x_clean = next(iter(ds_test.unbatch().batch(num_show)))

    xq = tf.cast(tf.round(x_noisy / in_scale + in_zero), tf.int8).numpy()

    yq_list = []
    for i in range(num_show):
        interpreter.set_tensor(in_det["index"], xq[i:i+1])
        interpreter.invoke()
        yq_list.append(interpreter.get_tensor(out_det["index"]))
    yq = np.concatenate(yq_list, axis=0)

    y_hat_int8 = (yq.astype(np.float32) - out_zero) * out_scale
    y_hat_int8 = np.clip(y_hat_int8, 0.0, 1.0)

    y_hat_float = model.predict(x_noisy, verbose=0)

    encoder_probe = tf.keras.Model(model.input, model.get_layer("bottleneck").output)
    z = encoder_probe.predict(x_noisy, verbose=0)

    def show_row(i):
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        ax[0].imshow(x_clean[i].numpy().squeeze(), cmap="gray"); ax[0].set_title("Clean target")
        ax[1].imshow(x_noisy[i].numpy().squeeze(), cmap="gray"); ax[1].set_title("Noisy input")
        ax[2].imshow(y_hat_int8[i].squeeze(), cmap="gray");      ax[2].set_title("Recon (INT8 TFLite)")
        ax[3].imshow(y_hat_float[i].squeeze(), cmap="gray");     ax[3].set_title("Recon (Float Keras)")
        for a in ax: a.axis("off")
        plt.show()

    for i in range(num_show):
        show_row(i)


if __name__ == "__main__":
    main()
