# Part I

## EEG Data Handling and Preprocessing
1. **Load EEG Data**: The `load_data` function reads EEG data from a MATLAB file (`.mat`). It extracts the sampling frequency (`fs`), EEG signals (`y`), and trigger signals (`trig`).

2. **Preprocess EEG Data**: The `preprocess_eeg_data` function performs several preprocessing steps:
   - Creates an MNE `info` object with channel names and types.
   - Sets up a standard 10-20 montage for EEG channels.
   - Initializes an MNE `RawArray` for handling EEG data.
   - Applies an average reference and bandpass filters the data between 1 Hz and 30 Hz.

3. **Manual Event Creation**: After preprocessing, the script manually identifies changes in the trigger channel to create events, which are used for epoching the data.

4. **Visualize Trigger Data**: It visualizes the entire trigger signal and the specific events detected to ensure they are correctly identified.

5. **Epoch Creation**: Using the identified events, the script creates epochs from the raw EEG data, defining a time window around each event.

6. **Artifact Rejection**: Implements a method to automatically determine rejection thresholds for artifact rejection in the epochs.

7. **ERP Calculation and Plotting**: Computes the average evoked response for different event types (target vs. non-target) and plots these.

8. **PSD Calculation**: For each epoch belonging to 'target' events, it calculates and plots the power spectral density.


# Part II

## Feature Extraction and Machine Learning
9. **Calculate PSD Features**: The `calculate_psd` function computes the power spectral density using FFT, which serves as a feature for machine learning.

10. **Prepare Data for Machine Learning**:
    - Labels are encoded numerically.
    - PSD features are calculated for each epoch.
    - Features are normalized using `StandardScaler`.
    - Addresses class imbalance by oversampling the minority class to balance the dataset.
    - Splits the balanced dataset into training and testing sets.

11. **Machine Learning Modeling**:
    - Trains a Random Forest classifier on the training data.
    - Evaluates the model on the test data and prints the accuracy and a detailed classification report.