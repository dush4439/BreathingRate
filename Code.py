import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert, find_peaks

# Load the initial CSV file
df = pd.read_csv('Data.csv')

# Define a function to perform moving average smoothing on data
def smooth_data(data, window_size=3):
    rolling_mean = data.rolling(window=window_size, min_periods=1).mean()
    return rolling_mean

def process_row(row):
    # Convert string representation of list to actual list
    amplitude_list = eval(row['Amplitude'])
    phase_list = eval(row['Phase'])

    # Apply estimation techniques
    amplitude_list = [np.nan if x == 0.0 else x for x in amplitude_list]
    phase_list = [np.nan if x == 0.0 else x for x in phase_list]
    interpolated_amplitude = pd.Series(amplitude_list).replace(0.0, np.nan).interpolate()
    interpolated_phase = pd.Series(phase_list).replace(0.0, np.nan).interpolate()
    if interpolated_amplitude.isna().any():
        interpolated_amplitude = interpolated_amplitude.ffill().bfill()
    if interpolated_phase.isna().any():
        interpolated_phase = interpolated_phase.ffill().bfill()

    # Apply smoothing
    smoothed_amplitude = smooth_data(interpolated_amplitude)
    smoothed_phase = smooth_data(interpolated_phase)

    # Return the processed lists as a series
    return pd.Series([smoothed_amplitude.tolist(), smoothed_phase.tolist()], index=['Amplitude', 'Phase'])

# Apply the processing function to each row
df[['Amplitude', 'Phase']] = df.apply(process_row, axis=1)

# Calculate the mean of each list in the 'Amplitude' and 'Phase' columns
df['Amplitude_Mean'] = df['Amplitude'].apply(lambda x: sum(x) / len(x))
df['Phase_Mean'] = df['Phase'].apply(lambda x: sum(x) / len(x))

# Apply the smoothing filter to the mean columns
df['Amplitude_Mean_Smooth'] = smooth_data(df['Amplitude_Mean'])
df['Phase_Mean_Smooth'] = smooth_data(df['Phase_Mean'])

# Drop unnecessary columns
df.drop(columns=['Amplitude', 'Phase', 'Amplitude_Mean', 'Phase_Mean'], inplace=True)

# Rename columns
df.rename(columns={'Amplitude_Mean_Smooth': 'Amplitude', 'Phase_Mean_Smooth': 'Phase'}, inplace=True)

df['Time'] = pd.to_datetime(df['Time'])

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Plot Amplitude vs. Time
axs[0].plot(df['Time'], df['Amplitude'], marker='o', color='blue')
axs[0].set_title('Amplitude over Time')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# Plot Phase vs. Time
axs[1].plot(df['Time'], df['Phase'], marker='x', color='red')
axs[1].set_title('Phase over Time')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Phase (radians)')
axs[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Design the Butterworth filter
N  = 4    # Filter order
Wn = 0.01  # Cutoff frequency
B, A = butter(N, Wn, output='ba')

# Apply the filter to your data
df['Amplitude'] = lfilter(B, A, df['Amplitude'])
df['Phase'] = lfilter(B, A, df['Phase'])
df.head(5)


# Calculate the moving average using a simple 'rolling window' technique
window_size = 3  # Adjust the window size as needed

# Drop rows with NaN values (optional)
df.dropna(subset=['Amplitude', 'Phase'], inplace=True)

# Calculate the moving average
df['Amplitude'] = df['Amplitude'].rolling(window=window_size, min_periods=1).mean()
df['Phase'] = df['Phase'].rolling(window=window_size, min_periods=1).mean()

# Display the first 5 rows to check the new columns with the filtered data
df.head(5)


df['Time'] = pd.to_datetime(df['Time'])

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Plot Amplitude vs. Time
axs[0].plot(df['Time'], df['Amplitude'], color='blue')
axs[0].set_title('Amplitude over Time')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# Plot Phase vs. Time
axs[1].plot(df['Time'], df['Phase'], color='red')
axs[1].set_title('Phase over Time')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Phase (radians)')
axs[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()



def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Calculate the Nyquist frequency (maximum frequency to avoid aliasing)
    low = lowcut / nyq  # Normalize lower cutoff frequency by the Nyquist frequency
    high = highcut / nyq  # Normalize upper cutoff frequency by the Nyquist frequency

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y



def calculate_breathing_rate(segment, fs):
    # Apply band-pass filter
    filtered_segment = bandpass_filter(segment, 0.1, 0.5, fs)

    # Obtain analytical signal
    analytical_signal = hilbert(filtered_segment)
    amplitude_envelope = np.abs(analytical_signal)

    # Detect peaks in the amplitude envelope
    peaks, _ = find_peaks(amplitude_envelope)

    # Calculate the time intervals between peaks
    peak_times = np.array(segment.index[peaks])
    time_intervals = np.diff(peak_times) / fs

    # Calculate breathing rate for the segment
    if len(time_intervals) > 0:
        average_breath_duration = np.mean(time_intervals)
        breathing_rate = 60 / average_breath_duration
    else:
        breathing_rate = 0

    return breathing_rate

# Assuming 'data' is a pandas DataFrame with your CSI data
# and 'fs' is the sampling frequency (e.g., 50 Hz)
fs = 10
breathing_rates = []
segment_duration = fs * 20  # 20-second segments

for i in range(0, len(df), segment_duration):
    segment = df['Amplitude'][i:i + segment_duration]
    if len(segment) == segment_duration:  # Ensure the segment is complete
        rate = calculate_breathing_rate(segment, fs)
        breathing_rates.append(rate)

# Calculate the average breathing rate over the recording period
average_breathing_rate = np.mean(breathing_rates) if breathing_rates else 0
print(f'Average Breathing Rate: {average_breathing_rate:.2f} breaths per minute')