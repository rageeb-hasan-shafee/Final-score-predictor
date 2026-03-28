from tensorflow import keras

# Load model
model = keras.models.load_model("ball_outcome_model.keras")

# Show model structure
model.summary()