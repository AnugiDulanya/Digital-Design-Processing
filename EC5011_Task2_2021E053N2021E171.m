% EC5011 - Task 02
% 2021/E/053 & 2021/E/171

% Step 1: Load and Analyze the Dataset
% Define the directories
train_dengue_dir = 'C:\Users\anugi\Desktop\EC5011_Task2\training\dengue\';
train_non_dengue_dir = 'C:\Users\anugi\Desktop\EC5011_Task2\training\non-dengue\';
test_dengue_dir = 'C:\Users\anugi\Desktop\EC5011_Task2\testing\dengue\';
test_non_dengue_dir = 'C:\Users\anugi\Desktop\EC5011_Task2\testing\non-dengue\';

% Read all audio files
train_dengue_audio = read_audio_files(train_dengue_dir);
train_non_dengue_audio = read_audio_files(train_non_dengue_dir);
[test_dengue_audio, total_test_dengue_files] = read_audio_files(test_dengue_dir);
[test_non_dengue_audio, total_test_non_dengue_files] = read_audio_files(test_non_dengue_dir);

% Display the number of audio files read
disp(['Number of dengue train files: ', num2str(length(train_dengue_audio))]);
disp(['Number of non-dengue train files: ', num2str(length(train_non_dengue_audio))]);
disp(['Number of dengue test files: ', num2str(total_test_dengue_files)]);
disp(['Number of non-dengue test files: ', num2str(total_test_non_dengue_files)]);

% Step 2: Frequency Domain Analysis
% Assume we process the first audio file as an example
train_dengue = train_dengue_audio{1};
train_non_dengue = train_non_dengue_audio{1};

% Compute FFT
D_train_fft = fft(train_dengue);
ND_train_fft = fft(train_non_dengue);

fs = 44100; % Assuming a sample rate of 44100 Hz
n_fft = length(D_train_fft); % Define a common FFT length for interpolation
f = (0:n_fft/2-1)*(fs/n_fft);

% Analyze all dengue train files
dengue_spectra = [];
for i = 1:length(train_dengue_audio)
    audio = train_dengue_audio{i};
    D_fft = abs(fft(audio, n_fft));
    dengue_spectra = [dengue_spectra; D_fft(1:n_fft/2)'];
end
mean_dengue_spectrum = mean(dengue_spectra, 1);

% Analyze all non-dengue train files
non_dengue_spectra = [];
for i = 1:length(train_non_dengue_audio)
    audio = train_non_dengue_audio{i};
    ND_fft = abs(fft(audio, n_fft));
    non_dengue_spectra = [non_dengue_spectra; ND_fft(1:n_fft/2)'];
end
mean_non_dengue_spectrum = mean(non_dengue_spectra, 1);

% Plot average spectra
figure;
subplot(2,1,1);
plot(f, mean_dengue_spectrum);
title('Average Dengue Mosquito Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 fs/2]); % Plot only up to Nyquist frequency

subplot(2,1,2);
plot(f, mean_non_dengue_spectrum);
title('Average Non-Dengue Mosquito Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 fs/2]); % Plot only up to Nyquist frequency

% Step 3: Design Filters
% Design bandpass filters based on the frequency analysis
dengue_band = [10 3150]; % Example frequency range
non_dengue_band = [3200 8250]; % Example frequency range

% Design filters
dengue_filter = designfilt('bandpassiir', 'FilterOrder', 8 ,'HalfPowerFrequency1', 10 , 'HalfPowerFrequency2', 3150 , 'SampleRate', fs);
non_dengue_filter = designfilt('bandpassiir', 'FilterOrder', 8 , 'HalfPowerFrequency1', 3200 , 'HalfPowerFrequency2', 8250 , 'SampleRate', fs);   

% Step 4: Feature Extraction and Classification
% Collect features from training data
train_features = [];
train_labels = [];

for i = 1:length(train_dengue_audio)
    audio = train_dengue_audio{i};
    energy_ratio = extract_features(audio, fs, dengue_filter, non_dengue_filter);
    train_features = [train_features; energy_ratio];
    train_labels = [train_labels; 1]; % 1 for dengue
end

for i = 1:length(train_non_dengue_audio)
    audio = train_non_dengue_audio{i};
    energy_ratio = extract_features(audio, fs, dengue_filter, non_dengue_filter);
    train_features = [train_features; energy_ratio];
    train_labels = [train_labels; 0]; % 0 for non-dengue
end

% Calculate mean energy ratios
dengue_features = train_features(train_labels == 1);
non_dengue_features = train_features(train_labels == 0);

mean_dengue_energy_ratio = mean(dengue_features);
mean_non_dengue_energy_ratio = mean(non_dengue_features);

disp(['Mean Dengue Energy Ratio: ', num2str(mean_dengue_energy_ratio)]);
disp(['Mean Non-Dengue Energy Ratio: ', num2str(mean_non_dengue_energy_ratio)]);

% Determine threshold value
threshold = 0.5 * (mean_dengue_energy_ratio + mean_non_dengue_energy_ratio);
disp(['Threshold Value: ', num2str(threshold)]);

% Train a classifier
classifier = fitcsvm(train_features, train_labels, 'KernelFunction', 'linear', 'Standardize', true);

% Classify test data
dengue_correct_classifications = 0;
non_dengue_correct_classifications = 0;

for i = 1:length(test_dengue_audio)
    test_audio = test_dengue_audio{i};
    energy_ratio = extract_features(test_audio, fs, dengue_filter, non_dengue_filter);
    prediction = predict(classifier, energy_ratio);
    
    if prediction == 1
        dengue_correct_classifications = dengue_correct_classifications + 1;
    end
end

for i = 1:length(test_non_dengue_audio)
    test_audio = test_non_dengue_audio{i};
    energy_ratio = extract_features(test_audio, fs, dengue_filter, non_dengue_filter);
    prediction = predict(classifier, energy_ratio);
    
    if prediction == 0
        non_dengue_correct_classifications = non_dengue_correct_classifications + 1;
    end
end

% Calculate accuracies
dengue_accuracy = dengue_correct_classifications / total_test_dengue_files;
non_dengue_accuracy = non_dengue_correct_classifications / total_test_non_dengue_files;
overall_accuracy = (dengue_correct_classifications + non_dengue_correct_classifications) / (total_test_dengue_files + total_test_non_dengue_files);

% Display accuracies
disp(['Dengue Accuracy: ', num2str(dengue_accuracy)]);
disp(['Non-Dengue Accuracy: ', num2str(non_dengue_accuracy)]);
disp(['Overall Accuracy: ', num2str(overall_accuracy)]);

% Plot the energy ratios for dengue and non-dengue
R_dengue = dengue_features; % Dengue energy ratios
R_nondengue = non_dengue_features; % Non-dengue energy ratios

% Initialize z for scatter plot (dummy variable for y-axis)
z_dengue = zeros(length(R_dengue), 1);
z_nondengue = zeros(length(R_nondengue), 1);

figure;
scatter(R_dengue, z_dengue, 'r');  % Plot dengue energy ratios in red
hold on;
scatter(R_nondengue, z_nondengue, 'g');  % Plot non-dengue energy ratios in green
title('Energy Ratio Scatter Plot');
xlabel('Energy Ratio');
ylabel('Dummy Variable');
legend('Dengue', 'Non-Dengue');
hold off;

% Function to read all audio files in a directory
function [audio_data, num_files] = read_audio_files(folder)
    files = dir(fullfile(folder, '*.wav'));
    num_files = length(files);
    audio_data = cell(num_files, 1); % Using cell array to store different lengths of audio
    for i = 1:num_files
        filename = fullfile(folder, files(i).name);
        [audio_data{i}, ~] = audioread(filename);
    end
end

% Function to extract features from audio
function energy_ratio = extract_features(audio, fs, dengue_filter, non_dengue_filter)
    chunk_size = fs; % 1-second chunks
    dengue_energy = 0;
    non_dengue_energy = 0;
    
    for start_idx = 1:chunk_size:length(audio)
        end_idx = min(start_idx + chunk_size - 1, length(audio));
        chunk = audio(start_idx:end_idx);
        
        % Filter the chunk
        chunk_dengue = filtfilt(dengue_filter, chunk);
        chunk_non_dengue = filtfilt(non_dengue_filter, chunk);
        
        % Compute energies
        chunk_energy_dengue = sum(chunk_dengue.^2);
        chunk_energy_non_dengue = sum(chunk_non_dengue.^2);
        
        dengue_energy = dengue_energy + chunk_energy_dengue;
        non_dengue_energy = non_dengue_energy + chunk_energy_non_dengue;
    end
    
    energy_ratio = dengue_energy / (dengue_energy + non_dengue_energy);
end
