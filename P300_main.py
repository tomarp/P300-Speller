import scipy.io as sio
import mne
from os.path import join
import matplotlib.pyplot as plt
from mne import pick_types
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(file_path):
    mat_data = sio.loadmat(file_path)
    fs = mat_data['fs'][0, 0]  # Sampling frequency
    y = mat_data['y']  # EEG data
    trig = mat_data['trig'][:, 0]  # Ensure trig is 1D
    return y, trig, fs

def preprocess_eeg_data(y, trig, fs):
    ch_names = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'Oz', 'PO8']
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    info.set_montage('standard_1020')
    raw = mne.io.RawArray(y.T, info)
    # set the average reference
    raw.set_eeg_reference('average', projection=False)
    raw.filter(1., 30., fir_design='firwin')
    return raw, trig

def plot_full_trigger_data(trig):
    plt.figure(figsize=(15, 5))
    plt.plot(trig, label='Trigger Signal')
    plt.title('Full Trigger Channel Data')
    plt.xlabel('Samples')
    plt.ylabel('Trigger Value')
    plt.legend()
    plt.show()

def reject_parameter(raw, events):
    picks = pick_types(raw.info, meg=True, eeg=True, stim=False)
    # Define event IDs
    event_id = {'non-target': -1, 'target': 1}
    dummy_epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.1, tmax=1.0,
                     picks=picks, baseline=(None, 0), preload=True,
                     reject=None)

    from autoreject import get_rejection_threshold  # noqa
    reject = get_rejection_threshold(dummy_epochs)
    return reject

def calculate_psd(data, sfreq):
    """Calculate power spectral density using numpy's FFT."""
    fft_data = np.fft.rfft(data, axis=2)
    psd = np.abs(fft_data) ** 2
    return psd.mean(axis=2)  # Average PSD across frequencies


def prepare_data(epochs):
    """Prepare labels and PSD features for machine learning."""
    labels = epochs.events[:, -1]
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    # Find indices of the minority and majority classes
    minority_indices = np.where(labels_encoded == 1)[0]
    majority_indices = np.where(labels_encoded == 0)[0]

    psd_mean = calculate_psd(epochs.get_data(), epochs.info['sfreq'])
    scaler = StandardScaler()
    psd_scaled = scaler.fit_transform(psd_mean)

    # Perform the manual oversampling process
    oversampled_minority_indices = np.random.choice(minority_indices, len(majority_indices), replace=True)
    resampled_indices = np.concatenate([majority_indices, oversampled_minority_indices])
    np.random.shuffle(resampled_indices)  # Shuffle indices to mix data

    features = psd_scaled[resampled_indices]
    labels = labels_encoded[resampled_indices]
    return train_test_split(features, labels, test_size=0.2, random_state=42)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train RandomForest and evaluate its performance."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['non-target', 'target'])

    return accuracy, report


# main function -->
# -------------------------------------------------------
if __name__ == '__main__':

    # ----------------------
    # Define path parameters
    # ----------------------
    root_dir = join('/', 'Users', 'diptyajit.das', 'Documents')
    subjects_dir = join(root_dir, 'p300', '')

    # subject ids
    ids = ['S1', 'S2', 'S3', 'S4', 'S5']

    accuracy_scores = []
    sub_reports = []

    # loop over the subject list
    for id in ids:
        # Load and preprocess the data
        file_path = join(subjects_dir, id+'.mat')

        y, trig, fs = load_data(file_path)
        raw, trig = preprocess_eeg_data(y, trig, fs)

        # Manually create events array from the cleaned triggers
        event_times = np.where(np.diff(trig) != 0)[0] + 1  # Find changes in trigger values
        event_values = trig[event_times]  # Get the trigger values at these times
        events = np.column_stack((event_times, np.zeros_like(event_times), event_values))

        # Plot full trigger data to understand its behavior
        plot_full_trigger_data(trig)

        # Define event IDs
        event_id = {'non-target': -1, 'target': 1}

        # Plot full trigger events for targets and non-targets
        fig = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_id)

        # Create epochs for ERP analysis
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.1, tmax=0.8, baseline=(None, 0), preload=True)

        # get reject parameter
        reject = reject_parameter(raw, events)
        cleaned_epochs = epochs.drop_bad(reject)

        evoked_trg = cleaned_epochs['target'].average()
        evoked_nontrg = cleaned_epochs['non-target'].average()

        evoked_trg.plot()
        evoked_nontrg.plot()

        # Plot the Power Spectral Density components
        psd_fig = cleaned_epochs['target'].plot_psd(show=True)
        plt.show()

        # Main execution --> train model
        X_train_fft, X_test_fft, y_train_fft, y_test_fft = prepare_data(epochs)
        model_accuracy, report = train_and_evaluate(X_train_fft, X_test_fft, y_train_fft, y_test_fft)

        accuracy_scores.append(model_accuracy)
        sub_reports.append(report)

    for i in range(len(ids)):
        print('Classifier results')
        print('subject:', ids[i])
        print('Accuracy:',  accuracy_scores[i])
        print('Classification Report:\n', sub_reports[i])
        print('-----------------------------------------')
