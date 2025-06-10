# This guide demonstrates how to use Keras for audio modeling tasks.
# It covers the essential steps from data preprocessing to model training and evaluation.
# We will be using a simple example of audio classification.

# --- Section 1: Setting up the Environment ---
# Before we begin, make sure you have the necessary libraries installed.
# You can install them using pip:
#
# pip install tensorflow librosa numpy matplotlib
#
# - tensorflow: For building and training neural networks.
# - librosa: For audio processing tasks like loading, feature extraction.
# - numpy: For numerical operations.
# - matplotlib: For plotting (optional, but useful for visualization).

import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# --- End of Section 1 ---

# --- Section 2: Loading and Preprocessing Audio Data ---
# In a real-world scenario, you would have a dataset of audio files.
# For this guide, we'll simulate loading and preprocessing.

# Helper function to load and preprocess a single audio file
def load_and_preprocess_audio(file_path, target_sr=22050, duration=5, n_mfcc=13):
    '''
    Loads an audio file, resamples it, extracts MFCCs, and pads/truncates.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sampling rate.
        duration (int): Desired duration of the audio in seconds.
        n_mfcc (int): Number of MFCCs to extract.

    Returns:
        np.ndarray: Processed MFCC features.
    '''
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None) # Load with original sampling rate

        # Resample if necessary
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Trim or pad audio to a fixed duration
        target_length = sr * duration
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)), 'constant')

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # (Optional) Normalize MFCCs (per file)
        # mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Example Usage (Illustrative) ---
# In a real task, you would loop through your dataset.
# For this example, let's imagine we have a list of file paths and labels.

# Note: You'll need actual audio files for this part to run.
# For demonstration, we'll create a dummy audio file.
# In a real scenario, replace this with paths to your audio dataset.
print("\n--- Example: Processing a Dummy Audio File ---")
try:
    # Create a dummy mono audio file for demonstration
    # This requires `soundfile` library: pip install soundfile
    import soundfile as sf
    dummy_sr = 22050
    dummy_duration = 3 # seconds
    dummy_frequency = 440 # Hz (A4 note)
    dummy_amplitude = 0.5
    t = np.linspace(0, dummy_duration, int(dummy_sr * dummy_duration), False)
    dummy_audio_data = dummy_amplitude * np.sin(2 * np.pi * dummy_frequency * t)
    dummy_file_path = "dummy_audio.wav"
    sf.write(dummy_file_path, dummy_audio_data, dummy_sr, subtype='PCM_16')
    print(f"Created a dummy audio file: {dummy_file_path}")

    # Process the dummy audio file
    example_mfccs = load_and_preprocess_audio(dummy_file_path, duration=3, n_mfcc=13)

    if example_mfccs is not None:
        print(f"Shape of extracted MFCCs: {example_mfccs.shape}") # (n_mfcc, time_frames)

        # Visualize the MFCCs
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(example_mfccs, sr=dummy_sr, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC')
        plt.tight_layout()
        # plt.show() # Uncomment to display plot if running locally
        plt.savefig("dummy_mfcc_visualization.png")
        print("Saved MFCC visualization to dummy_mfcc_visualization.png")
    else:
        print("Could not process the dummy audio file.")

except ImportError:
    print("Could not import 'soundfile'. Skipping dummy audio file creation and processing.")
    print("To run this part, please install it: pip install soundfile")
except Exception as e:
    print(f"An error occurred during dummy audio processing: {e}")


# --- Data Preparation for Model Training ---
# Assuming you have a list of processed features (e.g., MFCCs) and corresponding labels.
# X = [mfcc_features_audio1, mfcc_features_audio2, ...]
# y = [label_audio1, label_audio2, ...]

# For this guide, let's create some placeholder data.
# In a real scenario, this would come from your dataset.
num_samples = 100
num_classes = 5 # Example: 5 different sound categories
feature_height = 13 # n_mfcc
feature_width = 216 # Number of time frames, depends on duration and hop_length

# Generate random MFCC-like data
X_placeholder = np.random.rand(num_samples, feature_height, feature_width)
# Generate random labels (integer encoded)
y_placeholder = np.random.randint(0, num_classes, num_samples)

print(f"\nShape of placeholder features (X_placeholder): {X_placeholder.shape}")
print(f"Shape of placeholder labels (y_placeholder): {y_placeholder.shape}")

# Reshape features if needed by the model (e.g., for CNNs, add a channel dimension)
# For a CNN expecting (batch, height, width, channels):
X_placeholder_reshaped = X_placeholder[..., np.newaxis]
print(f"Shape of reshaped placeholder features for CNN: {X_placeholder_reshaped.shape}")

# Convert labels to one-hot encoding if using categorical_crossentropy
y_placeholder_one_hot = tf.keras.utils.to_categorical(y_placeholder, num_classes=num_classes)
print(f"Shape of one-hot encoded placeholder labels: {y_placeholder_one_hot.shape}")


# --- Splitting data into training and validation sets ---
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(
#     X_placeholder_reshaped, y_placeholder_one_hot, test_size=0.2, random_state=42
# )
# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
# print("Note: Using sklearn.model_selection.train_test_split requires scikit-learn: pip install scikit-learn")
# For now, let's manually split to avoid an extra dependency for this basic guide.
split_ratio = 0.8
split_index = int(num_samples * split_ratio)

X_train = X_placeholder_reshaped[:split_index]
y_train = y_placeholder_one_hot[:split_index]
X_val = X_placeholder_reshaped[split_index:]
y_val = y_placeholder_one_hot[split_index:]

print(f"\nManually split data:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# --- End of Section 2 ---

# --- Section 3: Building a Simple Audio Classification Model ---
# We'll build a Convolutional Neural Network (CNN) for audio classification.
# CNNs are effective for tasks involving grid-like data, such as MFCC spectrograms.

def build_audio_cnn_model(input_shape, num_classes):
    '''
    Builds a simple CNN model for audio classification.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (n_mfcc, time_frames, 1) for MFCCs).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled Keras model.
    '''
    model = tf.keras.models.Sequential([
        # Input Layer - Convolutional
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(), # Helps stabilize and speed up training

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),

        # Flatten the features before passing to Dense layers
        tf.keras.layers.Flatten(),

        # Dense Layers for Classification
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5), # Dropout for regularization

        # Output Layer
        tf.keras.layers.Dense(num_classes, activation='softmax') # Softmax for multi-class probability
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', # For one-hot encoded labels
                  metrics=['accuracy'])

    return model

# --- Define model parameters ---
# Input shape should match the preprocessed data: (height, width, channels)
# X_train.shape is (num_samples, feature_height, feature_width, 1)
input_shape_cnn = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

# Create the model instance
audio_cnn_model = build_audio_cnn_model(input_shape=input_shape_cnn, num_classes=num_classes)

# Print model summary
print("\n--- Audio CNN Model Summary ---")
audio_cnn_model.summary()

# --- End of Section 3 ---

# --- Section 4: Training the Model ---
# Now we train the model using our placeholder training data.

print("\n--- Training the Audio CNN Model ---")

# Training parameters:
#   epochs: One epoch is one pass through the entire training dataset.
#           Increase for real tasks (e.g., 20-100 or more, depending on dataset size and complexity).
#           Too few epochs can lead to underfitting; too many can lead to overfitting.
epochs = 2  # Using a small number for this guide for quick execution.
#   batch_size: Number of samples processed before the model's internal parameters are updated.
#               Affects memory usage and training speed. Common values are 32, 64, 128.
batch_size = 32

# Train the model using the .fit() method.
# Note: Training with random placeholder data won't result in a meaningful or accurate model.
# This step primarily demonstrates the training process itself.
history = audio_cnn_model.fit(
    X_train, y_train,              # Training data (features and labels)
    epochs=epochs,                 # Number of epochs to train for
    batch_size=batch_size,         # Number of samples per gradient update
    validation_data=(X_val, y_val),# Data on which to evaluate the loss and any model metrics
                                   # at the end of each epoch. This is crucial for monitoring overfitting.
    verbose=1                      # Controls the verbosity of output during training:
                                   # 0 = silent, 1 = progress bar, 2 = one line per epoch.
)

print("Model training finished (with placeholder data).")

# --- Plotting training history (Loss and Accuracy) ---
# The 'history' object returned by model.fit() contains training metrics
# (loss and accuracy) for each epoch, for both training and validation sets.
if history and history.history: # Check if history object and its .history attribute exist
    print("\n--- Plotting Training History ---")
    # Create a figure with two subplots (one for accuracy, one for loss)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8)) # (width, height) in inches

    # Plot training & validation accuracy values
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='lower right') # Show legend

    # Plot training & validation loss values
    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right') # Show legend

    plt.tight_layout() # Adjusts subplot params for a tight layout.
    # plt.show() # Uncomment to display plot if running interactively.
    plt.savefig("training_history.png") # Save the plot to a file.
    print("Saved training history plot to training_history.png")
else:
    print("Training history is not available to plot (e.g., if training was skipped or failed).")

# --- End of Section 4 ---

# --- Section 5: Evaluating the Model ---
# After training, it's crucial to evaluate the model's performance on data
# it has not seen during training. We use the validation set for this purpose.
# Ideally, one would also have a separate "test set" for a final, unbiased evaluation.

print("\n--- Evaluating the Model ---")

# Evaluate the model on the validation data using the .evaluate() method.
# This returns the loss value and metrics values for the model in test mode.
#   X_val: Validation features.
#   y_val: Validation labels (one-hot encoded).
#   verbose=0: Silent mode (no output during evaluation).
# The model should perform reasonably well on this data if it has generalized well.
# Poor performance might indicate overfitting to the training data.
loss, accuracy = audio_cnn_model.evaluate(X_val, y_val, verbose=0)

print(f"Validation Loss: {loss:.4f}") # The average loss over the validation set.
print(f"Validation Accuracy: {accuracy*100:.2f}%") # The accuracy on the validation set.

# --- Further Evaluation (Conceptual & Optional) ---
# For a more comprehensive evaluation, especially in real-world projects, consider:
#
# 1. Dedicated Test Set:
#    - A dataset completely held out during training and hyperparameter tuning.
#    - Provides the most unbiased measure of the model's generalization ability.
#
# 2. Confusion Matrix:
#    - A table showing the performance of a classification model.
#    - Rows represent true classes, columns represent predicted classes.
#    - Helps identify which classes are often confused with others.
#    - Example using scikit-learn (requires `pip install scikit-learn`):
#      from sklearn.metrics import confusion_matrix, classification_report
#      y_pred_probabilities = audio_cnn_model.predict(X_val) # Get raw probability outputs
#      y_pred_classes = np.argmax(y_pred_probabilities, axis=1) # Convert probabilities to class indices
#      y_true_classes = np.argmax(y_val, axis=1) # Convert one-hot y_val to class indices
#
#      conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
#      print("\nConfusion Matrix (Validation Set):\n", conf_matrix)
#
#      # Classification Report: Provides precision, recall, F1-score, and support per class.
#      print("\nClassification Report (Validation Set):\n",
#            classification_report(y_true_classes, y_pred_classes,
#                                  target_names=[f"Class {i}" for i in range(num_classes)]))
#      # (Note: `target_names` would ideally be actual class names like 'speech', 'music', etc.)
#
# 3. Precision, Recall, F1-score:
#    - Precision: TP / (TP + FP) - Ability of the classifier not to label a negative sample as positive.
#    - Recall (Sensitivity): TP / (TP + FN) - Ability of the classifier to find all positive samples.
#    - F1-score: 2 * (Precision * Recall) / (Precision + Recall) - Weighted average of Precision and Recall.
#    - These metrics are especially important for imbalanced datasets where accuracy alone can be misleading.
#
# 4. ROC Curve and AUC:
#    - For binary or multi-class (one-vs-all) classification.
#    - Receiver Operating Characteristic (ROC) curve plots True Positive Rate vs. False Positive Rate.
#    - Area Under the Curve (AUC) provides a single measure of separability.

# --- End of Section 5 ---

# --- Section 6: Making Predictions on New Audio Data ---
# Now, let's see how to use the trained model to make predictions on a new, unseen audio file.

print("\n--- Making Predictions on New Audio ---")

# We'll use the dummy audio file created in Section 2 as an example.
# In a real application, this would be any new audio file you want to classify.
new_audio_file_path = "dummy_audio.wav" # Defined in Section 2

if 'dummy_file_path' in globals() and new_audio_file_path == dummy_file_path:
    print(f"Attempting to make a prediction for: {new_audio_file_path}")

    # 1. Load and preprocess the new audio file.
    #    It is CRITICAL that the preprocessing steps (sampling rate, duration, n_mfcc, normalization if used)
    #    applied to this new audio file are IDENTICAL to those used for the training data.
    #    Our model was trained on features with shape (feature_height, feature_width),
    #    which are determined by n_mfcc and duration respectively.
    #    feature_height (n_mfcc) was X_train.shape[1] (e.g., 13).
    #    feature_width (time_frames) was X_train.shape[2] (e.g., 216).
    #    This width corresponds to a duration of approximately 5 seconds with librosa's default MFCC settings.
    #    So, we use duration=5 and n_mfcc=X_train.shape[1] (which is `feature_height`).

    # The comment below explains how the duration (5s) leads to feature_width (216 frames)
    #    To calculate expected duration for feature_width=216:
    #    Default hop_length for librosa.feature.mfcc is 512 samples.
    #    Target sampling rate (target_sr) is 22050 Hz.
    #    Number of time_frames = (target_sr * duration_in_seconds) / hop_length
    #    So, duration_in_seconds = (time_frames * hop_length) / target_sr
    #    expected_duration_for_216_frames = (216 * 512) / 22050 = 5.018... seconds.
    #    Thus, using duration=5 in `load_and_preprocess_audio` is appropriate to get ~216 frames.
    #    The exact number of frames can sometimes vary slightly due to rounding or boundary effects in librosa,
    #    so padding/truncation to the exact feature_width is important if there's a mismatch.
    #    For this guide, `load_and_preprocess_audio` already handles fixed duration processing.

    new_mfccs = load_and_preprocess_audio(
        new_audio_file_path,
        duration=5, # Must match the duration that results in `feature_width` (e.g., 216 frames)
        n_mfcc=X_train.shape[1] # Must match `feature_height` (e.g., 13 MFCCs)
    )

    if new_mfccs is not None:
        # Expected shape: (feature_height, feature_width), e.g., (13, 216)
        print(f"Shape of preprocessed MFCCs for new audio: {new_mfccs.shape}")

        # 2. Reshape the features to match the model's input shape.
        #    The CNN model expects input of shape (batch_size, height, width, channels).
        #    For a single prediction, batch_size is 1.
        #    Our `new_mfccs` is (height, width). We need to add batch and channel dimensions.

        # Ensure the processed features have the correct dimensions before reshaping.
        # input_shape_cnn is (height, width, channels), e.g., (13, 216, 1)
        if new_mfccs.shape[0] == input_shape_cnn[0] and new_mfccs.shape[1] == input_shape_cnn[1]:
            new_mfccs_reshaped = new_mfccs[np.newaxis, ..., np.newaxis]
            # Expected shape: (1, feature_height, feature_width, 1), e.g., (1, 13, 216, 1)
            print(f"Shape of reshaped MFCCs for model prediction: {new_mfccs_reshaped.shape}")

            # 3. Make a prediction using model.predict().
            #    This returns a list of probability distributions (one per sample in the batch).
            #    Since we have one sample, we take the first element.
            prediction_probabilities = audio_cnn_model.predict(new_mfccs_reshaped)[0] # Get probabilities for the single sample

            # The output `prediction_probabilities` is an array of probabilities for each class.
            # e.g., [0.1, 0.05, 0.7, 0.1, 0.05] for 5 classes.
            # Find the class index with the highest probability.
            predicted_class_index = np.argmax(prediction_probabilities)
            predicted_probability = prediction_probabilities[predicted_class_index]

            print(f"\nPrediction Results for '{new_audio_file_path}':")
            print(f"Predicted Class Index: {predicted_class_index}")
            # In a real application, you would map this index to a human-readable class name.
            # Example:
            # class_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E'] # Should match num_classes
            # if 0 <= predicted_class_index < len(class_names):
            #     print(f"Predicted Class Name: {class_names[predicted_class_index]}")
            # else:
            #     print("Predicted class index is out of bounds for class_names.")
            print(f"Confidence: {predicted_probability*100:.2f}%")

            # Optionally, display probabilities for all classes:
            # print("\nProbabilities for all classes:")
            # for i, prob in enumerate(prediction_probabilities):
            #     # class_name = class_names[i] if 0 <= i < len(class_names) else f"Class {i}"
            #     # print(f"  {class_name}: {prob*100:.2f}%")
        else:
            # This error handling is important for real applications.
            print(f"Error: The shape of the processed new audio ({new_mfccs.shape})")
            print(f"does not match the expected input dimensions ({input_shape_cnn[0]}, {input_shape_cnn[1]}) for the model.")
            print("Please ensure `duration` and `n_mfcc` in `load_and_preprocess_audio` for prediction")
            print("are consistent with the training data feature dimensions.")

    else:
        print(f"Could not preprocess the new audio file: {new_audio_file_path}")
elif 'dummy_file_path' not in globals() and 'sf' not in globals(): # Check if dummy file creation was skipped
    print("Dummy audio file was not created (likely 'soundfile' library is not installed or an error occurred).")
    print("Skipping prediction part as the example audio file is unavailable.")
else:
    # This case might occur if dummy_file_path was defined but the file wasn't found or some other issue.
    print(f"Dummy audio file '{new_audio_file_path}' not found or 'dummy_file_path' variable is not as expected.")
    print("Skipping prediction part.")


# --- Further Considerations & Best Practices ---
# - Class Labels / Names:
#   - Maintain a clear mapping from integer class indices (0, 1, 2...) to human-readable names
#     (e.g., `class_names = ['cat', 'dog', 'bird']`). This is essential for interpreting predictions.
#
# - Saving and Loading Models:
#   - For deploying or reusing your model without retraining, save it:
#     `audio_cnn_model.save("my_audio_classifier_model.h5")` # Saves in HDF5 format
#     `audio_cnn_model.save("my_audio_classifier_model_tf", save_format="tf")` # Saves in TensorFlow SavedModel format
#   - Load a saved model:
#     `from tensorflow.keras.models import load_model`
#     `loaded_model = load_model("my_audio_classifier_model.h5")` # Or path to SavedModel directory
#     `# loaded_model.summary()`
#     `# predictions = loaded_model.predict(new_data_reshaped)`
#
# - More Advanced Architectures:
#   - For complex audio tasks, explore:
#     - Recurrent Neural Networks (RNNs) like LSTMs or GRUs: Good for capturing temporal sequences.
#     - Attention Mechanisms: Can help the model focus on relevant parts of the audio.
#     - Hybrid CNN-RNN Models: Combine CNNs for feature extraction with RNNs for sequence modeling.
#     - Pre-trained Audio Models: Models like YAMNet, VGGish, or PANNs (if available and suitable)
#       can be used for transfer learning or as feature extractors, often yielding strong results.
#
# - Data Augmentation:
#   - If your dataset is small, data augmentation can help improve model generalization and reduce overfitting.
#   - Techniques for audio include:
#     - Time shifting: Slightly shift the audio left or right.
#     - Pitch shifting: Change the audio pitch.
#     - Adding background noise: Mix in noise from various sources.
#     - Time stretching: Speed up or slow down audio without changing pitch.
#     - Random gain/amplitude changes.
#   - Libraries like `audiomentations` can be helpful.
#
# - Hyperparameter Tuning:
#   - The performance of a neural network heavily depends on hyperparameters like:
#     - Learning rate, optimizer choice (Adam, SGD, etc.)
#     - Number of layers, number of units/filters per layer
#     - Activation functions, dropout rate, batch size
#   - Use techniques like Grid Search, Random Search, or Bayesian Optimization (e.g., with KerasTuner)
#     to find optimal hyperparameter combinations.
#
# - Handling Real-World Audio:
#   - Be prepared for variations in audio quality, noise levels, and recording conditions.
#   - Consider Voice Activity Detection (VAD) to remove silent parts if they are not relevant.
#   - Normalize volume levels across your dataset if they vary significantly.

print("\n--- End of Audio Modeling Guide ---")
# --- End of Section 6 ---
