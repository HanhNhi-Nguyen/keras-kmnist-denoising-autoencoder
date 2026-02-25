#train.py
import tensorflow as tf
from .data import make_datasets
from .model import build_autoencoder

def main():
    ds_train, ds_test, _ = make_datasets()

    model = build_autoencoder()
    model.summary()

    EPOCHS = 5
    history = model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS)
    print("Training done.")

    # Save trained float model for reuse by the other scripts
    model.save("results/float_model.keras")
    print("Saved float model: results/float_model.keras")

if __name__ == "__main__":
    main()
