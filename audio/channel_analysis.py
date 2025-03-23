import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import shutil


def analyze_channels(audio_file, tolerance=1e-6, dominance_threshold=0.8):
    """
    Analyzes the stereo channels in an audio file.

    Parameters:
    - audio_file: Path to the audio file
    - tolerance: Numerical tolerance for floating point comparison
    - dominance_threshold: Ratio of energy that determines channel dominance

    Returns:
    - Channel type classification
    """
    # Load audio file with stereo channels preserved
    y, sr = librosa.load(audio_file, mono=False)

    # Check if mono
    if y.shape[0] != 2:
        return "1ch"

    # Get left and right channels
    left_channel = y[0]
    right_channel = y[1]

    # Check if channels are identical
    if np.allclose(left_channel, right_channel, rtol=tolerance, atol=tolerance):
        return "2ch-identical"

    # Analyze channel energy distribution
    left_energy = np.sum(left_channel**2)
    right_energy = np.sum(right_channel**2)
    total_energy = left_energy + right_energy

    # Calculate energy ratios
    left_ratio = left_energy / total_energy
    right_ratio = right_energy / total_energy

    # Check for speaker separation by analyzing correlation and energy distribution
    correlation = np.corrcoef(left_channel, right_channel)[0, 1]

    # Check if one channel consistently dominates the other
    if left_ratio > dominance_threshold or right_ratio > dominance_threshold:
        return "2ch-unbalanced"  # One channel dominates - likely not separate speakers

    # Check if channels are different enough to likely contain different speakers
    # High correlation with balanced energy often means acoustic differences rather than different speakers
    if abs(correlation) > 0.7 and (0.3 < left_ratio < 0.7):
        return "2ch-acoustic"  # Different but highly correlated - likely acoustic differences

    # Low/moderate correlation with balanced energy often indicates different content like speakers
    return "2ch-distinct"  # Likely different speakers or content


def visualize_channel_comparison(audio_file, save_path=None) -> tuple[float, float, float, float]:
    """
    Creates a visualization of the two channels to help in analysis.

    Returns:
    - left_energy_mean: Mean energy of the left channel
    - right_energy_mean: Mean energy of the right channel
    - left_ratio: Ratio of left channel energy to total energy
    - right_ratio: Ratio of right channel energy to total energy
    """
    y, sr = librosa.load(audio_file, mono=False)

    if y.shape[0] != 2:
        print("Not a stereo file")
        return 0.0, 0.0, 0.0, 0.0

    left_channel = y[0]
    right_channel = y[1]

    plt.figure(figsize=(12, 8))

    # Plot waveforms
    plt.subplot(3, 1, 1)
    plt.plot(left_channel, label="Left channel", alpha=0.7)
    plt.plot(right_channel, label="Right channel", alpha=0.7)
    plt.legend()
    plt.title("Channel Waveforms")

    # Plot energy over time
    window_size = 1024
    hop_length = 512

    left_energy = np.array([np.sum(left_channel[i : i + window_size] ** 2) for i in range(0, len(left_channel) - window_size, hop_length)])

    right_energy = np.array([np.sum(right_channel[i : i + window_size] ** 2) for i in range(0, len(right_channel) - window_size, hop_length)])

    plt.subplot(3, 1, 2)
    plt.plot(left_energy, label="Left energy", alpha=0.7)
    plt.plot(right_energy, label="Right energy", alpha=0.7)
    plt.legend()
    plt.title("Channel Energy Over Time")

    # Plot channel difference
    plt.subplot(3, 1, 3)
    plt.plot(left_channel - right_channel)
    plt.title("Channel Difference")

    plt.tight_layout()

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    # Calculate energy ratios
    left_energy_mean = left_energy.mean()
    right_energy_mean = right_energy.mean()
    total_energy = left_energy_mean + right_energy_mean
    left_ratio = left_energy_mean / total_energy
    right_ratio = right_energy_mean / total_energy

    return left_energy_mean, right_energy_mean, left_ratio, right_ratio


def find_audio_files(directory):
    """Find all audio files in directory and its subdirectories"""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".wav", ".mp3", ".m4a")):
                audio_files.append(os.path.join(root, file))
    return audio_files


def get_user_input_for_rename(file_path, channel_type):
    """Get user input for file renaming"""
    print(f"\nFile: {file_path}")
    print(f"Detected channel type: {channel_type}")

    # Get number of speakers
    while True:
        num_speakers = input("Enter number of speakers (e.g., 1, 2, 3): ").strip()
        if num_speakers.isdigit() and int(num_speakers) > 0:
            break
        print("Please enter a valid positive number.")

    # Get languages
    while True:
        languages = input("Enter language codes separated by hyphen (e.g., en, en-fr, de): ").strip().lower()
        if languages and all(lang.isalpha() for lang in languages.split("-")):
            break
        print("Please enter valid language codes (e.g., en, fr, de).")

    # Get title
    while True:
        title = input("Enter a title for the file (e.g., KyotoTrain, AirFrance): ").strip()
        if title and all(c.isalnum() or c == "_" for c in title):
            break
        print("Please enter a valid title (alphanumeric characters and underscores only).")

    return num_speakers, languages, title


def rename_audio_file(file_path, channel_type):
    """Rename file according to convention and update related files"""
    # Get user input for renaming
    num_speakers, languages, title = get_user_input_for_rename(file_path, channel_type)

    # Construct new filename
    dir_path = os.path.dirname(file_path)
    file_ext = os.path.splitext(file_path)[1]
    new_filename = f"{channel_type}_{num_speakers}spk_{languages}_{title}{file_ext}"
    new_file_path = os.path.join(dir_path, new_filename)

    # Check if confirmation is needed
    if os.path.exists(new_file_path) and new_file_path != file_path:
        print(f"Warning: {new_filename} already exists!")
        return None

    # Confirm with user
    print(f"\nRename:")
    print(f"  From: {os.path.basename(file_path)}")
    print(f"  To:   {new_filename}")
    confirm = input("Confirm rename? (y/n): ").strip().lower()

    if confirm == "y":
        # Rename the audio file
        os.rename(file_path, new_file_path)

        # Find and rename related files (PNG, JSON, etc.)
        base_path = os.path.splitext(file_path)[0]
        for related_ext in [".png", ".json", ".ref.json"]:
            related_file = base_path + related_ext
            if os.path.exists(related_file):
                new_related_path = os.path.join(dir_path, f"{channel_type}_{num_speakers}spk_{languages}_{title}{related_ext}")
                os.rename(related_file, new_related_path)
                print(f"Also renamed: {os.path.basename(related_file)} -> {os.path.basename(new_related_path)}")

        return new_file_path
    else:
        print("Skipping rename.")
        return None


def already_follows_convention(file_path, channel_type):
    """Check if file already follows the naming convention"""
    filename = os.path.basename(file_path)
    return filename.startswith(f"{channel_type}_")


# Process all audio files in audio directory and subdirectories
audio_dir = "audio"
audio_files = find_audio_files(audio_dir)
results = {}
energy_data = {}
renamed_files = {}

print(f"Found {len(audio_files)} audio files to process.")

for file_path in audio_files:
    try:
        print(f"\nProcessing: {file_path}")
        channel_type = analyze_channels(file_path)
        results[file_path] = channel_type

        # Generate same directory structure for the PNG files
        rel_path = os.path.relpath(file_path, audio_dir)
        base_name = os.path.splitext(rel_path)[0]
        png_path = os.path.join(audio_dir, f"{base_name}.png")

        # Visualize the channels
        left_energy_mean, right_energy_mean, left_ratio, right_ratio = visualize_channel_comparison(file_path, save_path=png_path)

        # Store energy data for summary
        energy_data[file_path] = (left_energy_mean, right_energy_mean, left_ratio, right_ratio)

        # Check if file already follows naming convention
        if already_follows_convention(file_path, channel_type):
            print(f"File already follows naming convention with correct channel type: {channel_type}")
            continue

        # Ask user if they want to rename this file
        print(f"Channel type detected: {channel_type}")
        rename_this = input("Do you want to rename this file? (y/n): ").strip().lower()

        if rename_this == "y":
            new_path = rename_audio_file(file_path, channel_type)
            if new_path:
                renamed_files[file_path] = new_path

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# Update results and energy_data for renamed files
for old_path, new_path in renamed_files.items():
    if old_path in results:
        results[new_path] = results.pop(old_path)
    if old_path in energy_data:
        energy_data[new_path] = energy_data.pop(old_path)

# Summary
print("\nSummary:")
print(f"Total files processed: {len(audio_files)}")
print(f"Files renamed: {len(renamed_files)}")

for category in sorted(set(results.values())):
    files = [f for f, t in results.items() if t == category]
    print(f"\n{category}: {len(files)} files")
    for f in files:
        left_energy, right_energy, left_ratio, right_ratio = energy_data.get(f, (0, 0, 0, 0))
        print(f"  - {f} (Left: {left_energy:.2e}, Right: {right_energy:.2e}, L-ratio: {left_ratio:.2f}, R-ratio: {right_ratio:.2f})")
