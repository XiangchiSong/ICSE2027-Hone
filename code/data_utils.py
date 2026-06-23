import gzip
import numpy as np
import torch
import pickle
import os
from typing import Tuple, List, Optional, Union
from PIL import Image
import pandas as pd
import nltk
from collections import Counter
import re
import h5py
import glob
from skimage import io, color, exposure, transform

# Add custom NLTK data path
nltk.data.path.append(r"E:\Research\WWW2026\MIND_data\nltk_data")


class PyTorchDataFeeder:
    """
    Contains data as torch.tensors. Allows easy retrieval of data batches and
    in-built data shuffling.
    """

    def __init__(
        self, x, x_dtype, y, y_dtype, device, cast_device=None, transform=None
    ):
        """
        Return a new data feeder with copies of the input x and y data. Data is
        stored on device. If the intended model to use with the input data is
        on another device, then cast_device can be passed. Batch data will be
        sent to this device before being returned by next_batch. If transform is
        passed, the function will be applied to x data returned by next_batch.

        Args:
        - x:            x data to store
        - x_dtype:      torch.dtype or 'long'
        - y:            y data to store
        - y_dtype:      torch.dtype or 'long'
        - device:       torch.device to store data on
        - cast_device:  data from next_batch is sent to this torch.device
        - transform:    function to apply to x data from next_batch
        """
        if x_dtype == "long":
            self.x = torch.tensor(
                x, device=device, requires_grad=False, dtype=torch.int32
            ).long()
        else:
            self.x = torch.tensor(x, device=device, requires_grad=False, dtype=x_dtype)

        if y_dtype == "long":
            self.y = torch.tensor(
                y, device=device, requires_grad=False, dtype=torch.int32
            ).long()
        else:
            self.y = torch.tensor(y, device=device, requires_grad=False, dtype=y_dtype)

        self.idx = 0
        self.n_samples = x.shape[0]
        self.cast_device = cast_device
        self.transform = transform
        self.shuffle_data()

    def shuffle_data(self):
        """
        Co-shuffle x and y data.
        """
        ord = torch.randperm(self.n_samples)
        self.x = self.x[ord]
        self.y = self.y[ord]

    def next_batch(self, B):
        """
        Return batch of data If B = -1, the all data is returned. Otherwise, a
        batch of size B is returned. If the end of the local data is reached,
        the contained data is shuffled and the internal counter starts from 0.

        Args:
        - B:    size of batch to return

        Returns (x, y) tuple of torch.tensors. Tensors are placed on cast_device
                if this is not None, else device.
        """
        if B == -1:
            x = self.x
            y = self.y
            self.shuffle_data()

        elif self.idx + B > self.n_samples:
            # if batch wraps around to start, add some samples from the start
            extra = (self.idx + B) - self.n_samples
            x = torch.cat((self.x[self.idx :], self.x[:extra]))
            y = torch.cat((self.y[self.idx :], self.y[:extra]))
            self.shuffle_data()
            self.idx = extra

        else:
            x = self.x[self.idx : self.idx + B]
            y = self.y[self.idx : self.idx + B]
            self.idx += B

        if not self.cast_device is None:
            x = x.to(self.cast_device)
            y = y.to(self.cast_device)

        if not self.transform is None:
            x = self.transform(x)

        return x, y


# ============================================================================
# Private Raw Data Loaders
# ============================================================================


def _load_raw_mnist(
    data_dir: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load raw MNIST data from gzip files.

    Args:
        data_dir: Path to MNIST data directory

    Returns:
        ((x_train, y_train), (x_test, y_test)) as numpy arrays
    """
    train_x_fname = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_y_fname = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_x_fname = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_y_fname = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    # Load training data
    with gzip.open(train_x_fname) as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        x_train = x_train.astype(np.float32) / 255.0

    with gzip.open(train_y_fname) as f:
        y_train = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))

    # Load test data
    with gzip.open(test_x_fname) as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        x_test = x_test.astype(np.float32) / 255.0

    with gzip.open(test_y_fname) as f:
        y_test = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))

    return (x_train, y_train), (x_test, y_test)


def _load_raw_cifar10(
    data_dir: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load raw CIFAR-10 data from pickle files.

    Args:
        data_dir: Path to CIFAR-10 data directory

    Returns:
        ((x_train, y_train), (x_test, y_test)) as numpy arrays
    """
    fnames = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]

    # Create arrays to store all CIFAR-10 train data
    x_train = np.zeros((50000, 32, 32, 3), dtype=np.float32)
    y_train = np.zeros((50000), dtype=np.int32)

    # Load training batches
    for i, fname in enumerate(fnames):
        with open(os.path.join(data_dir, fname), "rb") as f:
            data_dict = pickle.load(f, encoding="bytes")

        images = data_dict[b"data"].reshape((10000, 32, 32, 3), order="F")
        images = np.rot90(images, k=3, axes=(1, 2)) / 255.0
        labels = np.array(data_dict[b"labels"])

        x_train[i * 10000 : (i + 1) * 10000, :, :, :] = images
        y_train[i * 10000 : (i + 1) * 10000] = labels

    # Load test data
    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    x_test = data_dict[b"data"].reshape((10000, 32, 32, 3), order="F")
    x_test = np.rot90(x_test, k=3, axes=(1, 2)) / 255.0
    y_test = np.array(data_dict[b"labels"])

    # Transpose to PyTorch format (N, C, H, W)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    return (x_train, y_train), (x_test, y_test)


def _load_raw_cifar100(
    data_dir: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load raw CIFAR-100 data from pickle files.

    Args:
        data_dir: Path to CIFAR-100 data directory

    Returns:
        ((x_train, y_train), (x_test, y_test)) as numpy arrays
    """
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    # Load training data
    train_path = os.path.join(data_dir, "train")
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"File {train_path} not found. Please check the path.")

    with open(train_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    images = data_dict[b"data"].reshape((50000, 3, 32, 32))
    y_train = np.array(data_dict[b"fine_labels"])

    # Normalize training data
    x_train = images / 255.0
    for c in range(3):
        x_train[:, c, :, :] = (
            x_train[:, c, :, :] - CIFAR100_TRAIN_MEAN[c]
        ) / CIFAR100_TRAIN_STD[c]

    # Load test data
    test_path = os.path.join(data_dir, "test")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"File {test_path} not found. Please check the path.")

    with open(test_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    images = data_dict[b"data"].reshape((10000, 3, 32, 32))
    y_test = np.array(data_dict[b"fine_labels"])

    # Normalize test data
    x_test = images / 255.0
    for c in range(3):
        x_test[:, c, :, :] = (
            x_test[:, c, :, :] - CIFAR100_TRAIN_MEAN[c]
        ) / CIFAR100_TRAIN_STD[c]

    return (x_train, y_train), (x_test, y_test)


def _load_raw_tinyimagenet(
    data_dir: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load raw TinyImageNet data from directory structure.

    Args:
        data_dir: Path to TinyImageNet data directory

    Returns:
        ((x_train, y_train), (x_test, y_test)) as numpy arrays
    """
    # Define paths
    base_path = os.path.join(data_dir, "tiny-imagenet-200")
    wnids_path = os.path.join(base_path, "wnids.txt")
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")

    # Build class label mapping
    with open(wnids_path, "r") as f:
        wnids = [line.strip() for line in f.readlines()]
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Load training data (100,000 images)
    x_train_list = []
    y_train_list = []

    for wnid in wnids:
        class_dir = os.path.join(train_path, wnid, "images")
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith(".JPEG"):
                    img_path = os.path.join(class_dir, img_name)
                    # Force convert to RGB to handle grayscale images
                    img = Image.open(img_path).convert("RGB")
                    img_array = np.array(img)
                    x_train_list.append(img_array)
                    y_train_list.append(wnid_to_label[wnid])

    # Load validation data as test set (10,000 images)
    val_annotations_path = os.path.join(val_path, "val_annotations.txt")
    val_img_to_label = {}

    with open(val_annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name = parts[0]
            wnid = parts[1]
            val_img_to_label[img_name] = wnid_to_label[wnid]

    x_test_list = []
    y_test_list = []

    val_images_path = os.path.join(val_path, "images")
    for img_name in os.listdir(val_images_path):
        if img_name.endswith(".JPEG") and img_name in val_img_to_label:
            img_path = os.path.join(val_images_path, img_name)
            # Force convert to RGB
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            x_test_list.append(img_array)
            y_test_list.append(val_img_to_label[img_name])

    # Convert to numpy arrays
    x_train = np.array(x_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    x_test = np.array(x_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int32)

    # Normalize pixel values to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Apply Z-score standardization with TinyImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Standardize each channel
    for c in range(3):
        x_train[:, :, :, c] = (x_train[:, :, :, c] - mean[c]) / std[c]
        x_test[:, :, :, c] = (x_test[:, :, :, c] - mean[c]) / std[c]

    # Transpose to PyTorch format (N, C, H, W)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    return (x_train, y_train), (x_test, y_test)


def _load_raw_gtsrb(
    data_dir: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Code Reference: Wu M, Wicker M, Ruan W, et al. A game-based approximate verification of deep neural networks with provable guarantees[J]. Theoretical Computer Science, 2020, 807: 298-329.
    Load raw GTSRB data with SOTA preprocessing and HDF5 caching.
    This function implements a self-contained GTSRB data loader with:
    - HDF5 caching for fast subsequent loads
    - SOTA image preprocessing (HSV histogram equalization)
    - Z-score standardization

    Args:
        data_dir: Path to GTSRB data directory

    Returns:
        ((x_train, y_train), (x_test, y_test)) as numpy arrays
    """
    # Define cache file path
    cache_path = os.path.join(data_dir, "gtsrb_processed_cache.h5")

    # Check if cache exists
    if os.path.exists(cache_path):
        # Fast path: Load from cache
        print("Loading GTSRB data from cache...")
        with h5py.File(cache_path, "r") as f:
            x_train = np.array(f["x_train"])
            y_train = np.array(f["y_train"])
            x_test = np.array(f["x_test"])
            y_test = np.array(f["y_test"])
    else:
        # Slow path: Process from raw data
        print("Processing GTSRB data from raw files...")

        # Define preprocessing function
        def _preprocess_gtsrb_image(img_array):

            # Convert to float
            img = img_array.astype(np.float32) / 255.0

            # Convert RGB to HSV
            img_hsv = color.rgb2hsv(img)

            # Apply histogram equalization to V channel
            img_hsv[:, :, 2] = exposure.equalize_hist(img_hsv[:, :, 2])

            # Convert back to RGB
            img_rgb = color.hsv2rgb(img_hsv)

            # Center crop to square
            h, w = img_rgb.shape[:2]
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2
            img_cropped = img_rgb[
                start_h : start_h + min_dim, start_w : start_w + min_dim
            ]

            # Resize to 48x48
            img_resized = transform.resize(img_cropped, (48, 48), anti_aliasing=True)

            return img_resized.astype(np.float32)

        # Process training data
        train_images_path = os.path.join(
            data_dir,
            "GTSRB_Final_Training_Images",
            "Final_Training",
            "Images",
            "*",
            "*.ppm",
        )
        train_paths = glob.glob(train_images_path)

        # Shuffle paths
        np.random.shuffle(train_paths)

        x_train_list = []
        y_train_list = []

        print(f"Processing {len(train_paths)} training images...")
        for i, img_path in enumerate(train_paths):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(train_paths)} images...")

            # Read image
            img = io.imread(img_path)

            # Preprocess
            img_processed = _preprocess_gtsrb_image(img)

            # Extract label from path (parent directory name)
            label = int(os.path.basename(os.path.dirname(img_path)))

            x_train_list.append(img_processed)
            y_train_list.append(label)

        # Process test data
        test_csv_path = os.path.join(data_dir, "GT-final_test.csv")
        test_df = pd.read_csv(test_csv_path, sep=";")

        x_test_list = []
        y_test_list = []

        print(f"Processing {len(test_df)} test images...")
        for idx, row in test_df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(test_df)} images...")

            # Build image path
            img_path = os.path.join(
                data_dir,
                "GTSRB_Final_Test_Images",
                "Final_Test",
                "Images",
                row["Filename"],
            )

            # Read image
            img = io.imread(img_path)

            # Preprocess
            img_processed = _preprocess_gtsrb_image(img)

            # Get label
            label = row["ClassId"]

            x_test_list.append(img_processed)
            y_test_list.append(label)

        # Convert to numpy arrays
        x_train = np.array(x_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.int32)
        x_test = np.array(x_test_list, dtype=np.float32)
        y_test = np.array(y_test_list, dtype=np.int32)

        # Save to cache
        print("Saving processed data to cache...")
        with h5py.File(cache_path, "w") as f:
            f.create_dataset("x_train", data=x_train, compression="gzip")
            f.create_dataset("y_train", data=y_train, compression="gzip")
            f.create_dataset("x_test", data=x_test, compression="gzip")
            f.create_dataset("y_test", data=y_test, compression="gzip")

    # Apply Z-score standardization
    print("Applying Z-score standardization...")

    # Calculate mean and std on training data only
    mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(x_train, axis=(0, 1, 2), keepdims=True)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    # Standardize both train and test sets
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Transpose to PyTorch format (N, C, H, W)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    print(f"GTSRB data loaded: train shape={x_train.shape}, test shape={x_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def _load_raw_urbansound8k(
    data_dir: str, test_fold: int = 10
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Code Reference: Zhang T, Feng T, Alam S, et al. Fedaudio: A federated learning benchmark for audio tasks[C]//ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023: 1-5.

    Load raw UrbanSound8K data with SOTA preprocessing and HDF5 caching.
    Converts audio files to log-mel spectrograms for CNN input.
    This function implements:
    - HDF5 caching for fast subsequent loads
    - SOTA audio preprocessing (mel-spectrograms)
    - Standardization and normalization
    - Proper 10-fold cross validation support as per official guidelines

    Args:
        data_dir: Path to UrbanSound8K data directory
        test_fold: Which fold to use as test set (1-10), following official protocol

    Returns:
        ((x_train, y_train), (x_test, y_test)) as numpy arrays
    """
    # Validate test_fold parameter
    if test_fold < 1 or test_fold > 10:
        raise ValueError(f"test_fold must be between 1 and 10, got {test_fold}")

    # Define cache file path with fold-specific naming
    cache_path = os.path.join(
        data_dir, f"urbansound8k_processed_cache_fold{test_fold}.h5"
    )

    # Check if cache exists
    if os.path.exists(cache_path):
        # Fast path: Load from cache
        print("Loading UrbanSound8K data from cache...")
        with h5py.File(cache_path, "r") as f:
            x_train = np.array(f["x_train"])
            y_train = np.array(f["y_train"])
            x_test = np.array(f["x_test"])
            y_test = np.array(f["y_test"])
    else:
        # Slow path: Process from raw data
        print("Processing UrbanSound8K data from raw files...")

        # Import librosa for audio processing
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for UrbanSound8K dataset. Please install it with: pip install librosa"
            )

        # Define audio preprocessing parameters (SOTA configuration)
        SAMPLE_RATE = 22050  # Standard sampling rate for UrbanSound8K
        DURATION = 4  # Uniform audio duration in seconds
        N_MELS = 128  # Number of mel frequency bands
        HOP_LENGTH = 512  # Frame shift
        N_FFT = 2048  # FFT window size
        SAMPLES_TO_CONSIDER = SAMPLE_RATE * DURATION

        # Load metadata
        metadata_path = os.path.join(
            data_dir, "UrbanSound8K", "metadata", "UrbanSound8K.csv"
        )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        print("Loading UrbanSound8K metadata...")
        metadata = pd.read_csv(metadata_path)

        # Initialize containers
        x_data_list = []
        y_data_list = []
        fold_list = []  # Keep track of folds for train/test split

        # Process each audio file
        total_files = len(metadata)
        print(f"Processing {total_files} audio files...")

        for idx, row in metadata.iterrows():
            if idx % 100 == 0:
                print(f"  Processed {idx}/{total_files} files...")

            # Build file path
            file_path = os.path.join(
                data_dir,
                "UrbanSound8K",
                "audio",
                f'fold{row["fold"]}',
                row["slice_file_name"],
            )

            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}, skipping...")
                continue

            try:
                # Load audio with librosa
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Ensure uniform length
                if len(signal) < SAMPLES_TO_CONSIDER:
                    # Pad with zeros if too short
                    signal = np.pad(
                        signal, (0, SAMPLES_TO_CONSIDER - len(signal)), mode="constant"
                    )
                else:
                    # Truncate if too long
                    signal = signal[:SAMPLES_TO_CONSIDER]

                # Extract mel spectrogram
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=signal,
                    sr=SAMPLE_RATE,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    n_mels=N_MELS,
                )

                # Convert to log scale (dB) with fixed reference
                log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0)

                # Add to data lists (we'll do global normalization later)
                x_data_list.append(log_mel_spectrogram)
                y_data_list.append(row["classID"])
                fold_list.append(row["fold"])

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}, skipping...")
                continue

        # Convert to numpy arrays
        x_data = np.array(x_data_list, dtype=np.float32)
        y_data = np.array(y_data_list, dtype=np.int32)
        fold_data = np.array(fold_list, dtype=np.int32)

        # Add channel dimension for CNN (N, 1, H, W)
        x_data = np.expand_dims(x_data, axis=1)

        # Apply global standardization (Z-score normalization)
        print("Applying global standardization to spectrograms...")
        mean = np.mean(x_data)
        std = np.std(x_data)
        if std > 0:
            x_data = (x_data - mean) / std
        else:
            print("Warning: Standard deviation is 0, skipping standardization")

        print(
            f"UrbanSound8K data processed: shape={x_data.shape}, classes={np.unique(y_data)}"
        )
        print(f"Data statistics: mean={np.mean(x_data):.4f}, std={np.std(x_data):.4f}")

        # Split into train/test using specified test_fold (following official 10-fold CV protocol)
        test_indices = np.where(fold_data == test_fold)[0]
        train_indices = np.where(fold_data != test_fold)[0]

        print(f"Using fold {test_fold} as test set (official UrbanSound8K protocol)")

        x_train = x_data[train_indices]
        y_train = y_data[train_indices]
        x_test = x_data[test_indices]
        y_test = y_data[test_indices]

        # Save to cache
        print("Saving processed data to cache...")
        with h5py.File(cache_path, "w") as f:
            f.create_dataset("x_train", data=x_train, compression="gzip")
            f.create_dataset("y_train", data=y_train, compression="gzip")
            f.create_dataset("x_test", data=x_test, compression="gzip")
            f.create_dataset("y_test", data=y_test, compression="gzip")

    print(
        f"UrbanSound8K data loaded: train shape={x_train.shape}, test shape={x_test.shape}"
    )

    return (x_train, y_train), (x_test, y_test)


def _load_raw_mind(
    data_dir: str,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray, dict
]:
    """
    Code Reference: Qi T, Wu F, Lyu L, et al. Fedsampling: A better sampling strategy for federated learning[J]. arXiv preprint arXiv:2306.14245, 2023.
    Load raw MIND news classification data from TSV files.

    This function implements SOTA text preprocessing for the MIND dataset:
    - Tokenization using NLTK
    - Vocabulary building with special tokens
    - GloVe embedding matrix construction
    - Sequence padding/truncation

    Args:
        data_dir: Path to MIND data directory

    Returns:
        ((x_train, y_train), (x_test, y_test), embedding_matrix, category_to_idx)
    """
    # Download NLTK punkt tokenizer if not already present
    try:
        # Try to find punkt_tab first (newer NLTK versions)
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            # Fallback to punkt (older NLTK versions)
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            # Try to download punkt_tab first, then punkt as fallback
            try:
                nltk.download("punkt_tab", quiet=True)
            except:
                nltk.download("punkt", quiet=True)

    # Define file paths
    train_news_path = os.path.join(data_dir, "MIND_train", "news.tsv")
    test_news_path = os.path.join(data_dir, "MIND_test", "news.tsv")

    # Check if files exist
    if not os.path.exists(train_news_path):
        raise FileNotFoundError(f"Training news file not found: {train_news_path}")
    if not os.path.exists(test_news_path):
        raise FileNotFoundError(f"Test news file not found: {test_news_path}")

    # Load training data
    print("Loading MIND training data...")
    train_df = pd.read_csv(
        train_news_path,
        sep="\t",
        header=None,
        names=[
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ],
    )

    # Load test data
    print("Loading MIND test data...")
    test_df = pd.read_csv(
        test_news_path,
        sep="\t",
        header=None,
        names=[
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ],
    )

    # Extract titles and categories
    train_titles = train_df["title"].fillna("").tolist()
    train_categories = train_df["category"].tolist()
    test_titles = test_df["title"].fillna("").tolist()
    test_categories = test_df["category"].tolist()

    # Build category mapping from training data
    unique_categories = sorted(list(set(train_categories)))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    num_classes = len(unique_categories)
    print(f"Found {num_classes} news categories: {unique_categories}")

    # Convert categories to indices
    y_train = np.array(
        [category_to_idx[cat] for cat in train_categories], dtype=np.int32
    )
    y_test = np.array(
        [category_to_idx.get(cat, 0) for cat in test_categories], dtype=np.int32
    )  # Default to 0 for unknown categories

    # Text preprocessing
    print("Preprocessing text data...")
    max_seq_len = 30  # SOTA parameter for news titles

    # Tokenize all titles
    def tokenize_text(text):
        """Tokenize and clean text"""
        # Basic cleaning
        text = text.lower().strip()
        # Remove special characters but keep spaces
        text = re.sub(r"[^\w\s]", " ", text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        return tokens

    train_tokenized = [tokenize_text(title) for title in train_titles]
    test_tokenized = [tokenize_text(title) for title in test_titles]

    # Build vocabulary from training data
    print("Building vocabulary...")
    word_freq = Counter()
    for tokens in train_tokenized:
        word_freq.update(tokens)

    # Create vocabulary with special tokens
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}

    # Add words with frequency >= 2 (common practice to reduce vocabulary size)
    min_freq = 2
    for word, freq in word_freq.items():
        if freq >= min_freq and word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # Convert tokens to indices with padding/truncation
    def tokens_to_indices(tokens, word_to_idx, max_len):
        """Convert tokens to indices with padding/truncation"""
        indices = []
        for token in tokens[:max_len]:  # Truncate if too long
            indices.append(word_to_idx.get(token, word_to_idx["<UNK>"]))

        # Pad if too short
        while len(indices) < max_len:
            indices.append(word_to_idx["<PAD>"])

        return indices

    # Convert all sequences
    x_train = np.array(
        [
            tokens_to_indices(tokens, word_to_idx, max_seq_len)
            for tokens in train_tokenized
        ],
        dtype=np.int32,
    )
    x_test = np.array(
        [
            tokens_to_indices(tokens, word_to_idx, max_seq_len)
            for tokens in test_tokenized
        ],
        dtype=np.int32,
    )

    # Build embedding matrix with GloVe
    print("Building embedding matrix with GloVe...")
    embedding_dim = 300  # GloVe 300d
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    # Try to load GloVe embeddings
    glove_path = r"E:\Research\WWW2026\MIND_data\glove.840B.300d.txt"
    if not os.path.exists(glove_path):
        # Try alternative path
        glove_path = os.path.join(os.path.dirname(data_dir), "glove.840B.300d.txt")

    if os.path.exists(glove_path):
        print(f"Loading GloVe embeddings from {glove_path}...")
        glove_embeddings = {}

        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                if len(values) > 300:  # Ensure valid embedding
                    word = " ".join(values[:-300])  # Handle multi-word tokens
                    vector = np.array(values[-300:], dtype=np.float32)
                    glove_embeddings[word] = vector

        # Fill embedding matrix
        embedded_words = 0
        for word, idx in word_to_idx.items():
            if word in glove_embeddings:
                embedding_matrix[idx] = glove_embeddings[word]
                embedded_words += 1
            elif word != "<PAD>":  # Keep <PAD> as zeros
                # Initialize with small random values for OOV words
                embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)

        print(
            f"Found GloVe embeddings for {embedded_words}/{vocab_size} words ({embedded_words/vocab_size*100:.1f}%)"
        )
    else:
        print(f"Warning: GloVe file not found at {glove_path}")
        print("Initializing embeddings with random values...")
        # Initialize all non-PAD embeddings with small random values
        for idx in range(1, vocab_size):  # Skip PAD at index 0
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)

    return (x_train, y_train), (x_test, y_test), embedding_matrix, category_to_idx


# ============================================================================
# Core Partitioning Engine
# ============================================================================


def _partition_data_by_dirichlet(
    x_data: np.ndarray,
    y_data: np.ndarray,
    num_clients: int,
    iid_label: float,
    iid_data: float,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Partition data using Dirichlet distribution for label skew and Log-Normal for quantity skew.

    This implements a state-of-the-art federated learning data partitioning strategy that allows
    fine-grained control over both label heterogeneity and quantity heterogeneity.

    Args:
        x_data: Feature data array
        y_data: Label data array
        num_clients: Number of clients to partition data among
        iid_label: Label heterogeneity parameter (0.0-1.0). 0.0=min heterogeneity, 1.0=max heterogeneity
        iid_data: Data quantity heterogeneity parameter (0.0-1.0). 0.0=balanced, 1.1=max imbalance
        seed: Random seed for reproducibility

    Returns:
        (x_client_list, y_client_list): Lists of data arrays for each client
    """
    if seed is not None:
        np.random.seed(seed)

    # Input validation and type safety
    assert (
        isinstance(num_clients, int) and num_clients > 0
    ), "num_clients must be a positive integer"
    assert len(x_data) == len(y_data), "x_data and y_data must have the same length"

    # Ensure y_data is integer type for safe indexing
    y_data = np.asarray(y_data, dtype=np.int32)

    # Clamp parameters to [0, 1] range and convert to distribution parameters
    iid_label = np.clip(iid_label, 0.0, 1.0)
    iid_data = np.clip(iid_data, 0.0, 1.0)

    # Convert iid_label (0-1) to Dirichlet alpha parameter
    # 0.0 -> 1000.0 (very homogeneous), 1.0 -> 0.1 (very heterogeneous)
    alpha_label = 0.1 + (1000.0 - 0.1) * (1.0 - iid_label)

    # Convert iid_data (0-1) to Log-Normal sigma parameter
    # 0.0 -> 0.0 (balanced), 1.0 -> 2.0 (very imbalanced)
    sigma_data = 2.0 * iid_data

    n_samples = len(y_data)
    unique_labels = np.unique(y_data)
    n_classes = len(unique_labels)

    # Ensure labels are contiguous integers starting from 0
    if not np.array_equal(unique_labels, np.arange(n_classes)):
        # Create label mapping for non-contiguous labels
        label_map = {
            old_label: new_label for new_label, old_label in enumerate(unique_labels)
        }
        y_data = np.array([label_map[label] for label in y_data], dtype=np.int32)

    # Step 1: Determine data quantity distribution using Log-Normal
    if sigma_data == 0.0:
        # Balanced distribution - use integer arithmetic
        base_samples = n_samples // num_clients
        samples_per_client = np.full(num_clients, base_samples, dtype=np.int32)

        # Distribute remaining samples
        remaining = n_samples % num_clients
        if remaining > 0:
            samples_per_client[:remaining] += 1
    else:
        # Log-Normal distribution for imbalanced quantities
        avg_samples = float(n_samples) / num_clients
        raw_samples = np.random.lognormal(
            mean=np.log(avg_samples), sigma=sigma_data, size=num_clients
        )

        # Normalize and convert to integers with careful rounding
        normalized_samples = raw_samples * n_samples / np.sum(raw_samples)
        samples_per_client = np.round(normalized_samples).astype(np.int32)

        # Ensure samples are at least 1
        samples_per_client = np.maximum(samples_per_client, 1)

        # Adjust for rounding errors using integer arithmetic
        total_assigned = np.sum(samples_per_client)
        diff = n_samples - total_assigned

        if diff > 0:
            # Add samples to clients with fewest samples
            indices = np.argsort(samples_per_client)[:diff]
            samples_per_client[indices] += 1
        elif diff < 0:
            # Remove samples from clients with most samples (ensure no negative)
            indices = np.argsort(samples_per_client)[::-1]  # Descending order
            for i in range(-diff):
                if samples_per_client[indices[i]] > 0:
                    samples_per_client[indices[i]] -= 1

    # Verify total samples match
    assert (
        np.sum(samples_per_client) == n_samples
    ), f"Sample allocation error: {np.sum(samples_per_client)} != {n_samples}"

    # Step 2: Generate label distributions using Dirichlet
    # Each row represents a client's label distribution
    label_distributions = np.random.dirichlet(
        [alpha_label] * n_classes, size=num_clients
    )

    # Step 3: Create class-wise indices with type safety
    class_indices = {}
    for c in range(n_classes):
        indices = np.where(y_data == c)[0].astype(np.int32)
        np.random.shuffle(indices)
        class_indices[c] = indices

    # Track how many samples of each class have been allocated
    class_allocated = np.zeros(n_classes, dtype=np.int32)

    # Step 4: Allocate data to clients with enhanced type safety
    x_clients = []
    y_clients = []

    for client_id in range(num_clients):
        # Calculate target number of samples per class for this client
        client_samples = int(samples_per_client[client_id])
        client_label_dist = label_distributions[client_id]

        # Target samples per class using stable integer arithmetic
        raw_targets = client_samples * client_label_dist
        target_samples_per_class = np.round(raw_targets).astype(np.int32)

        # Ensure targets sum to client_samples (handle rounding errors)
        target_sum = np.sum(target_samples_per_class)
        if target_sum != client_samples:
            diff = client_samples - target_sum
            if diff > 0:
                # Add to classes with highest fractional parts
                fractional_parts = raw_targets - target_samples_per_class
                indices = np.argsort(fractional_parts)[::-1][:diff]
                target_samples_per_class[indices] += 1
            elif diff < 0:
                # Remove from classes with lowest fractional parts
                fractional_parts = raw_targets - target_samples_per_class
                indices = np.argsort(fractional_parts)[:(-diff)]
                target_samples_per_class[indices] = np.maximum(
                    target_samples_per_class[indices] - 1, 0
                )

        # Collect indices for this client
        client_indices = []

        # First pass: allocate based on target distribution
        for c in range(n_classes):
            target = int(target_samples_per_class[c])
            available = len(class_indices[c]) - class_allocated[c]
            actual = min(target, available)

            if actual > 0:
                start_idx = int(class_allocated[c])
                end_idx = start_idx + actual
                selected_indices = class_indices[c][start_idx:end_idx]
                client_indices.extend(selected_indices.tolist())
                class_allocated[c] += actual

        # Second pass: fill remaining slots if needed
        current_total = len(client_indices)
        if current_total < client_samples:
            remaining_needed = client_samples - current_total

            # Find classes with remaining samples
            available_classes = []
            for c in range(n_classes):
                available = len(class_indices[c]) - class_allocated[c]
                if available > 0:
                    available_classes.append((c, available))

            # Sort by availability (descending)
            available_classes.sort(key=lambda x: x[1], reverse=True)

            for c, available in available_classes:
                if remaining_needed <= 0:
                    break

                take = min(remaining_needed, available)
                start_idx = int(class_allocated[c])
                end_idx = start_idx + take
                selected_indices = class_indices[c][start_idx:end_idx]
                client_indices.extend(selected_indices.tolist())
                class_allocated[c] += take
                remaining_needed -= take

        # Convert to numpy array with explicit integer type
        client_indices = np.array(client_indices, dtype=np.int32)

        # Verify indices are valid
        assert (
            len(client_indices) <= client_samples
        ), f"Too many indices for client {client_id}"
        assert np.all(client_indices >= 0) and np.all(
            client_indices < len(x_data)
        ), "Invalid indices detected"

        # Shuffle to mix classes
        np.random.shuffle(client_indices)

        # Extract client data
        x_clients.append(x_data[client_indices])
        y_clients.append(y_data[client_indices])

    return x_clients, y_clients


# ============================================================================
# Dataset Loading Interface
# ============================================================================


def load_federated_dataset(
    dataset_name: str,
    data_dir: str,
    num_clients: int,
    iid_label: float,
    iid_data: float,
    user_test: bool = True,
    seed: Optional[int] = None,
    test_fold: Optional[int] = None,
) -> Tuple[
    Tuple[List[np.ndarray], List[np.ndarray]],
    Tuple[Union[List[np.ndarray], np.ndarray], Union[List[np.ndarray], np.ndarray]],
]:
    """
    Unified interface for loading and partitioning federated datasets.

    This function provides a single entry point for all dataset loading and partitioning,
    implementing state-of-the-art Dirichlet-based partitioning for federated learning.

    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', 'cifar100', 'tinyimagenet', 'urbansound8k')
        data_dir: Path to dataset directory
        num_clients: Number of clients to partition data among
        iid_label: Label heterogeneity parameter (0.0-1.0). 0.0=min heterogeneity, 1.0=max heterogeneity
        iid_data: Data quantity heterogeneity parameter (0.0-1.0). 0.0=balanced, 1.0=max imbalance
        user_test: If True, partition test data among clients; if False, return global test set
        seed: Random seed for reproducibility
        test_fold: For UrbanSound8K only - which fold to use as test set (1-10). If None, uses fold 10

    Returns:
        ((x_train_clients, y_train_clients), (x_test, y_test))
        where train data is always partitioned, and test data format depends on user_test
    """
    # Load raw data
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = _load_raw_mnist(data_dir)
    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = _load_raw_cifar10(data_dir)
    elif dataset_name == "cifar100":
        (x_train, y_train), (x_test, y_test) = _load_raw_cifar100(data_dir)
    elif dataset_name == "tinyimagenet":
        (x_train, y_train), (x_test, y_test) = _load_raw_tinyimagenet(data_dir)
    elif dataset_name == "gtsrb":
        (x_train, y_train), (x_test, y_test) = _load_raw_gtsrb(data_dir)
    elif dataset_name == "urbansound8k":
        # Use specified test_fold or default to fold 10
        fold_to_use = test_fold if test_fold is not None else 10
        (x_train, y_train), (x_test, y_test) = _load_raw_urbansound8k(
            data_dir, test_fold=fold_to_use
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Partition training data
    x_train_clients, y_train_clients = _partition_data_by_dirichlet(
        x_train, y_train, num_clients, iid_label, iid_data, seed=seed
    )

    # Handle test data
    if user_test:
        # Partition test data among clients with same distribution as training
        x_test_clients, y_test_clients = _partition_data_by_dirichlet(
            x_test, y_test, num_clients, iid_label, iid_data, seed=seed
        )
        test_data = (x_test_clients, y_test_clients)
    else:
        # Return global test set (wrapped in lists for consistency)
        test_data = ([x_test], [y_test])

    return (x_train_clients, y_train_clients), test_data


def load_mind_federated_dataset(
    data_dir: str,
    num_clients: int,
    iid_label: float,
    iid_data: float,
    user_test: bool,
    seed: Optional[int] = None,
) -> Tuple[
    Tuple[List[np.ndarray], List[np.ndarray]],
    Tuple[Union[List[np.ndarray], np.ndarray], Union[List[np.ndarray], np.ndarray]],
    np.ndarray,
    int,
]:
    """
    Load and partition MIND dataset for federated learning.

    This function serves as the main entry point for MIND dataset processing,
    handling all text-specific preprocessing and returning additional information
    needed for the Text-CNN model.

    Args:
        data_dir: Path to MIND data directory
        num_clients: Number of clients to partition data among
        iid_label: Label heterogeneity parameter (0.0-1.0)
        iid_data: Data quantity heterogeneity parameter (0.0-1.0)
        user_test: If True, partition test data among clients
        seed: Random seed for reproducibility

    Returns:
        ((x_train_clients, y_train_clients), test_data, embedding_matrix, num_classes)
        where test_data format depends on user_test parameter
    """
    # Load raw MIND data with text preprocessing
    (
        (x_train, y_train),
        (x_test, y_test),
        embedding_matrix,
        category_to_idx,
    ) = _load_raw_mind(data_dir)

    # Get number of classes
    num_classes = len(category_to_idx)

    # Partition training data using existing engine
    x_train_clients, y_train_clients = _partition_data_by_dirichlet(
        x_train, y_train, num_clients, iid_label, iid_data, seed=seed
    )

    # Handle test data
    if user_test:
        # Partition test data among clients
        x_test_clients, y_test_clients = _partition_data_by_dirichlet(
            x_test, y_test, num_clients, iid_label, iid_data, seed=seed
        )
        test_data = (x_test_clients, y_test_clients)
    else:
        # Return global test set
        test_data = ([x_test], [y_test])

    return (x_train_clients, y_train_clients), test_data, embedding_matrix, num_classes


# ============================================================================
# Utility Functions (kept for compatibility)
# ============================================================================


def add_noise_to_frac(xs, frac, std):
    """
    Random 0-mean gaussian noise with given std will be added to (frac*len(xs))
    of the arrays in xs. Noisy values are clipped between 0-1.

    Args:
        - xs:   (list)      containing numpy ndarrays to add noise to
        - frac: (float) 0   <= frac <= 1, fraction of xs to add noise to
        - std:  (float)     standard deviation of noise.

    Returns:
        Tuple containing (noisy copy of xs, indexes of noisy vals)
    """
    idxs = np.random.choice(len(xs), int(len(xs) * frac), replace=False)

    new_xs = []
    for i in range(len(xs)):
        if i in idxs:
            noisy = xs[i] + np.random.normal(0.0, std, size=xs[i].shape)
            new_xs.append(np.clip(noisy, 0.0, 1.0))
        else:
            new_xs.append(np.copy(xs[i]))

    return new_xs, idxs


def to_tensor(x, device, dtype):
    """
    Convert Numpy array to torch.tensor.

    Args:
    - x:        (np.ndarray)   array to convert
    - device:   (torch.device) to place tensor on
    - dtype:    (torch.dtype)  or 'long' to convert to pytorch long
    """
    if dtype == "long":
        return torch.tensor(
            x, device=device, requires_grad=False, dtype=torch.int32
        ).long()
    else:
        return torch.tensor(x, device=device, requires_grad=False, dtype=dtype)


# ============================================================================
# Legacy Functions (for backward compatibility - will be deprecated)
# ============================================================================


def load_mnist(data_dir, W, iid, user_test=False):
    """Legacy function - use load_federated_dataset instead"""
    # Map old iid values to new 0-1 range parameters
    if iid == 1:  # balanced-iid
        iid_label, iid_data = 1.0, 0.0  # min heterogeneity, balanced
    elif iid == 2:  # balanced-non-iid
        iid_label, iid_data = 0.3, 0.0  # moderate heterogeneity, balanced
    elif iid == 3:  # imbalanced-non-iid
        iid_label, iid_data = 0.3, 0.5  # moderate heterogeneity, moderate imbalance
    elif iid == 4:  # imbalanced-mixed-iid
        iid_label, iid_data = 0.7, 0.5  # low heterogeneity, moderate imbalance
    else:
        raise ValueError(f"Unknown iid value: {iid}")

    return load_federated_dataset("mnist", data_dir, W, iid_label, iid_data, user_test)


def load_cifar10(data_dir, W, iid, user_test=False):
    """Legacy function - use load_federated_dataset instead"""
    # Map old iid values to new 0-1 range parameters
    if iid == 1:  # balanced-iid
        iid_label, iid_data = 1.0, 0.0  # min heterogeneity, balanced
    elif iid == 2:  # balanced-non-iid
        iid_label, iid_data = 0.3, 0.0  # moderate heterogeneity, balanced
    elif iid == 3:  # imbalanced-non-iid
        iid_label, iid_data = 0.3, 0.5  # moderate heterogeneity, moderate imbalance
    elif iid == 4:  # imbalanced-mixed-iid
        iid_label, iid_data = 0.7, 0.5  # low heterogeneity, moderate imbalance
    else:
        raise ValueError(f"Unknown iid value: {iid}")

    return load_federated_dataset(
        "cifar10", data_dir, W, iid_label, iid_data, user_test
    )


def load_cifar100(data_dir, W, iid, user_test=False):
    """Legacy function - use load_federated_dataset instead"""
    # Map old iid values to new 0-1 range parameters
    if iid == 1:  # balanced-iid
        iid_label, iid_data = 1.0, 0.0  # min heterogeneity, balanced
    elif iid == 2:  # balanced-non-iid
        iid_label, iid_data = 0.3, 0.0  # moderate heterogeneity, balanced
    elif iid == 3:  # imbalanced-non-iid
        iid_label, iid_data = 0.3, 0.5  # moderate heterogeneity, moderate imbalance
    elif iid == 4:  # imbalanced-mixed-iid
        iid_label, iid_data = 0.7, 0.5  # low heterogeneity, moderate imbalance
    else:
        raise ValueError(f"Unknown iid value: {iid}")

    return load_federated_dataset(
        "cifar100", data_dir, W, iid_label, iid_data, user_test
    )
