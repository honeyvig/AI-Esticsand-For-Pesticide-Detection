# AI-Esticsand-For-Pesticide-Detection
estiscand, is a handheld device designed to detect pesticide residues on produce using a non-destructive method. It employs spectrophotometry, which involves measuring how light of various wavelengths is reflected off the surface of fruits and vegetables. A machine learning model then analyzes this data to determine the presence of pesticides. Pestiscand consists of a sensor, a power supply, a display screen, and a processor. During testing, the device achieved an accuracy rate of identifying pesticide residues on spinach and tomatoes of greater than 85%, meeting the project's objectives for effectiveness and speed. 
----------
To create a Python program for a handheld device like EstiScand, which uses spectrophotometry to detect pesticide residues on produce, we can break the task down into the following components:

    Sensor Data Acquisition: The device uses a sensor to collect spectrophotometric data. We would simulate this step since hardware interaction is needed for actual sensor readings.

    Preprocessing the Sensor Data: Spectrophotometric data needs to be preprocessed before it's fed into a machine learning model. This may involve normalizing the data, filtering noise, and extracting relevant features.

    Machine Learning Model: A trained machine learning model, such as a classification model, would take the processed sensor data and predict whether pesticide residues are present.

    User Interface: We would simulate the device's display that shows results to the user. In this case, we can use a terminal-based output to simulate the display screen.

Step-by-Step Breakdown:

    Data Collection and Preprocessing:
        Spectrophotometry data will be simulated as a set of wavelength and intensity readings.
        Feature engineering might include normalizing the intensity data, filtering noise, and extracting key wavelength peaks.

    Model Training:
        We could use a dataset with labeled examples of whether pesticide residues are present (e.g., a set of labeled spinach and tomato data).
        A classification model, such as Random Forest or SVM, could be trained on this data.

    Prediction:
        For new data, the trained model would predict the presence of pesticide residues.

    Display the Results:
        The device will display whether pesticides are detected with a probability score.

Below is a Python program that simulates the entire EstiScand system:
Step 1: Simulating Sensor Data (Spectrophotometry)

import numpy as np
import random
import time

# Simulating the spectral data collection
def collect_spectrophotometry_data():
    # Simulating a spectral scan from a sensor with wavelengths from 200nm to 1000nm
    wavelengths = np.arange(200, 1001, 10)  # wavelengths between 200nm and 1000nm
    intensities = np.array([random.uniform(0.1, 1.0) for _ in wavelengths])
    
    # Add random noise to simulate real-world sensor imperfections
    noise = np.random.normal(0, 0.05, len(intensities))  # Gaussian noise
    intensities += noise

    # Return simulated spectral data
    return wavelengths, intensities

# Function to simulate the process of collecting data
def collect_data_for_testing():
    print("Collecting spectrophotometry data for analysis...")
    wavelengths, intensities = collect_spectrophotometry_data()
    print(f"Spectral Data (Wavelengths: {len(wavelengths)} samples):")
    print(f"Wavelengths: {wavelengths[:10]}...")  # Show first 10 wavelengths
    print(f"Intensities: {intensities[:10]}...")  # Show first 10 intensity values
    return wavelengths, intensities

Step 2: Feature Extraction and Preprocessing

# Simple data preprocessing function to normalize the intensities
def preprocess_data(intensities):
    max_intensity = np.max(intensities)
    normalized_intensities = intensities / max_intensity  # Normalize to 0-1 range
    return normalized_intensities

# Simulate extracting relevant features from the sensor data
def extract_features(wavelengths, intensities):
    normalized_intensities = preprocess_data(intensities)
    
    # For simplicity, we can use a simple feature: average intensity in a certain wavelength range
    feature_vector = np.mean(normalized_intensities[50:150])  # Example: average intensity between 500-600nm
    return np.array([feature_vector])

Step 3: Machine Learning Model (Train a Model)

Here we'll use a simple Random Forest Classifier to simulate the pesticide residue detection process. Normally, you would train this model on a labeled dataset of spectrophotometric data.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulating a training dataset
def generate_training_data():
    # For simplicity, we simulate some "normal" data and "pesticide" data
    X = []  # Features (normalized intensity averages)
    y = []  # Labels (1: pesticide detected, 0: no pesticide)

    # Simulate 100 samples
    for _ in range(100):
        wavelengths, intensities = collect_spectrophotometry_data()
        features = extract_features(wavelengths, intensities)
        
        # Randomly assign whether pesticide is detected (1 or 0)
        label = 1 if random.random() > 0.85 else 0  # 85% chance of no pesticide
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)

# Train the model on the simulated dataset
def train_model():
    X, y = generate_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model

Step 4: Predicting Pesticide Residues

Now that we have the model, let's use it to predict whether pesticide residues are detected on new produce.

def predict_pesticide_residues(model, wavelengths, intensities):
    features = extract_features(wavelengths, intensities)
    
    # Predict the presence of pesticides (0: no pesticide, 1: pesticide detected)
    prediction = model.predict([features])[0]
    confidence = model.predict_proba([features])[0][prediction]  # Get the confidence of the prediction
    
    return prediction, confidence

# Example of running the EstiScand prediction system
def run_estiscand():
    print("Training EstiScand model...")
    model = train_model()
    
    # Simulate collecting new sensor data for a test sample (e.g., spinach or tomato)
    wavelengths, intensities = collect_data_for_testing()

    print("Predicting pesticide residues...")
    prediction, confidence = predict_pesticide_residues(model, wavelengths, intensities)

    # Show result on device's display
    if prediction == 1:
        print(f"Warning: Pesticide residues detected with {confidence * 100:.2f}% confidence.")
    else:
        print(f"No pesticide residues detected with {confidence * 100:.2f}% confidence.")

Step 5: Running the EstiScand Device

To run the EstiScand system, simply call the run_estiscand() function.

if __name__ == "__main__":
    run_estiscand()

Explanation:

    Data Simulation:
        collect_spectrophotometry_data(): Simulates the collection of spectrophotometric data from the sensor, with random noise added.
        extract_features(): Extracts features from the spectral data, which could involve normalization and calculating averages over specific wavelength ranges.

    Model Training:
        The model is trained on simulated data using a RandomForestClassifier. The target variable y represents whether pesticides are detected (1 for pesticide and 0 for no pesticide).

    Prediction:
        The trained model is used to predict whether pesticide residues are present on a new sample of produce.
        The prediction is made by extracting features from the spectrophotometric data and then feeding them into the model.

    Display:
        The output (prediction and confidence) is printed to the console, simulating the device’s display screen.

Conclusion:

This Python code simulates the EstiScand handheld device for detecting pesticide residues using spectrophotometry and machine learning. In a real-world scenario, you would replace the simulated sensor data with actual sensor readings, possibly using libraries for hardware integration, and the model would be trained on a much larger, real-world dataset of spectrophotometric readings labeled with pesticide presence.
