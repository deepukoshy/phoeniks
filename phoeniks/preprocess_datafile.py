import numpy as np


class ProcessedData:
    def __init__(self, time, ref_data_td, sample_data_td, noise_floor_td, freq, ref_data_std_real_fd,
                 ref_data_std_imag_fd, sample_data_std_real_fd, sample_data_std_imag_fd, noise_floor_std_real_fd,
                 noise_floor_std_imag_fd):
        self.time = time
        self.ref_data_td = ref_data_td
        self.sample_data_td = sample_data_td
        self.noise_floor_td = noise_floor_td
        self.freq = freq
        self.ref_data_std_real_fd = ref_data_std_real_fd
        self.ref_data_std_imag_fd = ref_data_std_imag_fd
        self.sample_data_std_real_fd = sample_data_std_real_fd
        self.sample_data_std_imag_fd = sample_data_std_imag_fd
        self.noise_floor_std_real_fd = noise_floor_std_real_fd
        self.noise_floor_std_imag_fd = noise_floor_std_imag_fd



def preprocess_data(input_file, num_points=100, use_zero_padding=False, power_of_2=None):
    # Load the data from the input file, skipping the first row (headers)
    data = np.loadtxt(input_file, delimiter='\t', skiprows=1)

    # Extract the columns
    time = data[:, 0]
    ref_data_td = data[:, 1:4]
    sample_data_td = data[:, 4:7]

    # Calculate the time interval
    time_interval = time[1] - time[0]

    # Calculate the number of samples
    num_samples = len(time)

    # Perform zero-padding if requested
    if use_zero_padding and power_of_2 is not None:
        next_power_of_2 = 2 ** power_of_2
        if next_power_of_2 >= num_samples:
            padded_samples = next_power_of_2 - num_samples

            # Perform zero-padding for the reference data
            ref_data_td = np.pad(ref_data_td, [(0, padded_samples), (0, 0)], mode='constant')

            # Perform zero-padding for the sample data
            sample_data_td = np.pad(sample_data_td, [(0, padded_samples), (0, 0)], mode='constant')

            # Update the number of samples
            num_samples = next_power_of_2

    # Calculate the averaged time domain reference data and sample data
    ref_data_avg_td = np.mean(ref_data_td, axis=1)
    sample_data_avg_td = np.mean(sample_data_td, axis=1)

    # Calculate the standard deviation of the time domain reference data and sample data
    ref_data_std = np.std(ref_data_td, axis=1)
    sample_data_std = np.std(sample_data_td, axis=1)

    # Calculate the noise floor multi array
    noise_floor_multi_td = np.empty_like(ref_data_td)
    for i in range(ref_data_td.shape[1]):
        noise_floor_multi_td[:, i] = np.random.choice(ref_data_td[:num_points, i], size=num_samples, replace=True)

    # Calculate averaged noise floor
    noise_floor_td = np.mean(noise_floor_multi_td, axis=1)

    # Calculate the standard deviation of the time domain noise floor
    noise_floor_std_td = np.std(noise_floor_multi_td, axis=1)

    # Calculate the frequency domain standard deviations using FFT
    ref_data_freq_std = np.fft.fft(ref_data_std, n=num_samples)
    ref_data_freq_std = np.fft.fftshift(ref_data_freq_std)

    sample_data_freq_std = np.fft.fft(sample_data_std, n=num_samples)
    sample_data_freq_std = np.fft.fftshift(sample_data_freq_std)

    noise_floor_freq_std = np.fft.fft(noise_floor_std_td, n=num_samples)
    noise_floor_freq_std = np.fft.fftshift(noise_floor_freq_std)

    # Calculate the frequency values
    freq = np.fft.fftfreq(num_samples, d=time_interval)
    freq = np.fft.fftshift(freq)

    # Return the generated/calculated data as a processed_data object
    return ProcessedData(time, ref_data_avg_td, sample_data_avg_td, noise_floor_td, freq,
                         ref_data_freq_std.real, ref_data_freq_std.imag,
                         sample_data_freq_std.real, sample_data_freq_std.imag, noise_floor_freq_std.real,
                         noise_floor_freq_std.imag)
