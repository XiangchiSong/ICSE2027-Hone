import torch
import numpy as np
import operator
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F


class FLModel(torch.nn.Module):
    """
    Has methods to easily allow setting of model parameters, performing training
    steps, and retrieving model parameters.
    """

    def __init__(self, device):
        """
        Args:
            - device: (torch.device) to place model on
        """
        super(FLModel, self).__init__()
        self.optim = None
        self.device = device
        self.loss_fn = None
        self.bn_layers = []  # BN layers must be added here with any model

    def set_optim(self, optim, init_optim=True):
        """
        Set the optimizer that this model will perform SGD with.

        Args:
            - optim:     (torch.optim) that model will perform SGD steps with
            - init_optim (bool)        whether to initialise optimiser params
        """
        self.optim = optim
        if init_optim:
            self.empty_step()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise
        optimiser parameters.
        """
        raise NotImplementedError()

    def get_params(self):
        """
        Returns model values as NumpyModel. BN layer statistics (mu, sigma) are
        added as parameters at the end of the returned model.
        """
        ps = [np.copy(p.data.cpu().numpy()) for p in list(self.parameters())]
        for bn in self.bn_layers:
            ps.append(np.copy(bn.running_mean.cpu().numpy()))
            ps.append(np.copy(bn.running_var.cpu().numpy()))

        return NumpyModel(ps)

    def get_bn_vals(self, setting=0):
        """
        Returns the parameters from BN layers. Setting can be one of the
        following {0: usyb, 1: yb, 2: us, 3: none} to get different types of
        parameters.

        Args:
            - setting: (int) BN values to return

        Returns:
            list of [np.ndarrays] containing BN parameters
        """
        if setting not in [0, 1, 2, 3]:
            raise ValueError("Setting must be in: {0, 1, 2, 3}")

        vals = []

        if setting == 3:
            return vals

        with torch.no_grad():
            # add gamma, beta
            if setting in [0, 1]:
                for bn in self.bn_layers:
                    vals.append(np.copy(bn.weight.cpu().numpy()))
                    vals.append(np.copy(bn.bias.cpu().numpy()))

            # add mu, sigma
            if setting in [0, 2]:
                for bn in self.bn_layers:
                    vals.append(np.copy(bn.running_mean.cpu().numpy()))
                    vals.append(np.copy(bn.running_var.cpu().numpy()))
        return vals

    def set_bn_vals(self, vals, setting=0):
        """
        Set the BN parameterss of the model. Setting can be one of the following
        {0: usyb, 1: yb, 2: us, 3: none}.

        Args:
            - vals:     (NumpyModel) new BN values to set
            - setting:  (int)        type of values to return
        """
        if setting not in [0, 1, 2, 3]:
            raise ValueError("Setting must be in: {0, 1, 2, 3}")

        if setting == 3:
            return

        with torch.no_grad():
            i = 0
            # set gamma, beta
            if setting in [0, 1]:
                for bn in self.bn_layers:
                    bn.weight.copy_(torch.tensor(vals[i]))
                    bn.bias.copy_(torch.tensor(vals[i + 1]))
                    i += 2

            # set mu, sigma
            if setting in [0, 2]:
                for bn in self.bn_layers:
                    bn.running_mean.copy_(torch.tensor(vals[i]))
                    bn.running_var.copy_(torch.tensor(vals[i + 1]))
                    i += 2

    def set_params(self, params):
        """
        Passed params should be in the order of the model layers, as returned by
        get_params(), with the BN layer statistics (mu, sigma) appended to the
        end of the model.

        Args:
            - params:   (NumpyModel) to set model values with
        """
        i = 0
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.tensor(params[i]))
                i += 1

            #     # Reshape params[i] to match the shape of p before copying
            #     reshaped_param = torch.tensor(params[i]).view_as(p) # much more robust method, and will work for any shape of p
            #     p.copy_(reshaped_param)
            #     i += 1

            # set mu, sigma again, since the params in the BN layers will be changed in the training process
            for bn in self.bn_layers:
                bn.running_mean.copy_(torch.tensor(params[i]))
                bn.running_var.copy_(torch.tensor(params[i + 1]))
                i += 2

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs
        """
        raise NotImplementedError()

    def calc_acc(self, logits, y):
        """
        Calculate accuracy/performance metric of model.

        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values

        Returns:
            torch.tensor containing scalar value.
        """
        raise NotImplementedError()

    def train_step(self, x, y):
        """
        Perform one step of SGD using assigned optimizer.

        Args:
            - x: (torch.tensor) inputs
            - y: (torch.tensor) targets

        Returns:
            tupe of floats (loss, acc) calculated during the training step.
        """
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.calc_acc(logits, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item(), acc

    def test(self, x, y, B):
        """
        Calculate error and accuracy of passed data using batches of size B.

        Args:
            - x: (torch.tensor) inputs
            - y: (torch.tensor) labels
            - B: (int)          batch size

        Returns:
            tuple of floats (loss, acc) averaged over passed data.
        """
        self.eval()
        n_batches = int(np.ceil(x.shape[0] / B))
        loss = 0.0
        acc = 0.0

        with torch.no_grad():
            for b in range(n_batches):
                logits = self.forward(x[b * B : (b + 1) * B])
                loss += self.loss_fn(logits, y[b * B : (b + 1) * B]).item()
                acc += self.calc_acc(logits, y[b * B : (b + 1) * B])
        self.train()

        return loss / n_batches, acc / n_batches


class CIFAR100Model(FLModel):
    """
    ResNet-18 model adapted for CIFAR-100 dataset (32x32 images, 100 classes).
    Uses modified ResNet-18 architecture optimized for smaller input size.
    """

    def __init__(self, device):
        """
        Initialize CIFAR100Model with ResNet-18 architecture adapted for 32x32 images.

        Args:
            - device: (torch.device) device to place model on
        """
        super(CIFAR100Model, self).__init__(device)

        # Load ResNet-18 model without pretrained weights
        resnet18_model = models.resnet18(weights=None)

        # SOTA adaptations for 32x32 images
        # 1. Modify first convolutional layer: smaller kernel and stride for CIFAR-100
        resnet18_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 2. Remove first max pooling layer to preserve spatial resolution
        resnet18_model.maxpool = nn.Identity()

        # 3. Modify classifier for 100 classes
        num_ftrs = resnet18_model.fc.in_features
        resnet18_model.fc = nn.Linear(num_ftrs, 100)

        self.model = resnet18_model.to(device)

        # Collect all BatchNorm layers for framework compatibility
        self.bn_layers = [
            module
            for module in self.model.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            - x: (torch.tensor) input tensor, must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 100)
        """
        return self.model(x)

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalized predictions
            - y:      (torch.tensor) true labels

        Returns:
            torch.tensor containing scalar accuracy value
        """
        _, preds = torch.max(logits, dim=1)
        correct = (preds == y).sum().float()
        acc = correct / y.size(0)
        return acc

    def empty_step(self):
        """
        Perform one step of SGD with dummy data to initialize optimizer parameters.
        """
        # Create dummy input and target for CIFAR-100 (32x32 images)
        dummy_input = torch.zeros((2, 3, 32, 32), device=self.device)
        dummy_target = torch.zeros((2), dtype=torch.long, device=self.device)

        # Perform one training step
        self.train_step(dummy_input, dummy_target)


class MNISTModel(FLModel):
    """
    2-hidden-layer fully connected model, 2 hidden layers with 200 units and a
    BN layer. Categorical Cross Entropy loss.
    """

    def __init__(self, device):
        """
        Returns a new MNISTModelBN.

        Args:
            - device: (torch.device) to place model on
        """
        super(MNISTModel, self).__init__(device)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        self.fc0 = torch.nn.Linear(784, 200).to(device)
        self.relu0 = torch.nn.ReLU().to(device)

        self.fc1 = torch.nn.Linear(200, 200).to(device)
        self.relu1 = torch.nn.ReLU().to(device)

        self.out = torch.nn.Linear(200, 10).to(device)

        self.bn0 = torch.nn.BatchNorm1d(200).to(device)

        self.bn_layers = [self.bn0]

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        bn_insert = self.bn0(self.relu0(self.fc0(x)))
        result = self.relu1(self.fc1(bn_insert))

        return self.out(result)

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values

        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialse
        optimiser parameters.
        """
        self.train_step(
            torch.zeros((2, 784), device=self.device, dtype=torch.float32),
            torch.zeros((2), device=self.device, dtype=torch.int32).long(),
        )


class CIFAR10Model(FLModel):
    """
    Convolutional model with two (Conv -> ReLU -> MaxPool -> BN) blocks, and one
    fully connected hidden layer. Categorical Cross Entropy loss.
    """

    def __init__(self, device):
        """
        Returns a new CIFAR10Model.

        Args:
            - device: (torch.device) to place model on
        """
        super(CIFAR10Model, self).__init__(device)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1).to(device)
        self.relu0 = torch.nn.ReLU().to(device)
        self.pool0 = torch.nn.MaxPool2d(2, 2).to(device)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1).to(device)
        self.relu1 = torch.nn.ReLU().to(device)
        self.pool1 = torch.nn.MaxPool2d(2, 2).to(device)

        self.flat = torch.nn.Flatten().to(device)
        self.fc0 = torch.nn.Linear(2304, 512).to(device)
        self.relu2 = torch.nn.ReLU().to(device)

        self.out = torch.nn.Linear(512, 10).to(device)

        self.bn0 = torch.nn.BatchNorm2d(32).to(device)
        self.bn1 = torch.nn.BatchNorm2d(64).to(device)

        self.bn_layers = [self.bn0, self.bn1]

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        bn_insert1 = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        bn_insert2 = self.bn1(self.pool1(self.relu1(self.conv1(bn_insert1))))
        result = self.relu2(self.fc0(self.flat(bn_insert2)))

        return self.out(result)

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values

        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise
        optimiser parameters.
        """
        self.train_step(
            torch.zeros((2, 3, 32, 32), device=self.device, dtype=torch.float32),
            torch.zeros((2), device=self.device, dtype=torch.int32).long(),
        )


class TinyImageNetModel(FLModel):
    """
    ResNet-50 model adapted for TinyImageNet dataset (64x64 images, 200 classes).
    Uses modified ResNet-50 architecture optimized for smaller input size.
    """

    def __init__(self, device):
        """
        Initialize TinyImageNetModel with ResNet-50 architecture adapted for 64x64 images.

        Args:
            - device: (torch.device) device to place model on
        """
        super(TinyImageNetModel, self).__init__(device)

        # Load ResNet-50 model without pretrained weights
        resnet50_model = models.resnet50(weights=None)

        # SOTA adaptations for 64x64 images
        # 1. Modify first convolutional layer: smaller kernel and stride
        resnet50_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 2. Remove first max pooling layer to preserve spatial resolution
        resnet50_model.maxpool = nn.Identity()

        # 3. Modify classifier for 200 classes
        num_ftrs = resnet50_model.fc.in_features
        resnet50_model.fc = nn.Linear(num_ftrs, 200)

        self.model = resnet50_model.to(device)

        # Collect all BatchNorm layers for framework compatibility
        self.bn_layers = [
            module
            for module in self.model.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            - x: (torch.tensor) input tensor, must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 200)
        """
        return self.model(x)

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalized predictions
            - y:      (torch.tensor) true labels

        Returns:
            torch.tensor containing scalar accuracy value
        """
        _, preds = torch.max(logits, dim=1)
        correct = (preds == y).sum().float()
        acc = correct / y.size(0)
        return acc

    def empty_step(self):
        """
        Perform one step of SGD with dummy data to initialize optimizer parameters.
        """
        # Create dummy input and target for TinyImageNet (64x64 images)
        dummy_input = torch.zeros((2, 3, 64, 64), device=self.device)
        dummy_target = torch.zeros((2), dtype=torch.long, device=self.device)

        # Perform one training step
        self.train_step(dummy_input, dummy_target)


class GTSRBModel(FLModel):
    """
    ResNet-18 model adapted for GTSRB dataset (48x48 images, 43 classes).
    Uses modified ResNet-18 architecture optimized for smaller input size.
    Memory-efficient alternative using same architecture as CIFAR100Model.
    """

    def __init__(self, device):
        """
        Initialize GTSRBModel with ResNet-18 architecture adapted for 48x48 images.

        Args:
            - device: (torch.device) device to place model on
        """
        super(GTSRBModel, self).__init__(device)

        # Load ResNet-18 model without pretrained weights (same as CIFAR100Model)
        resnet18_model = models.resnet18(weights=None)

        # SOTA adaptations for 48x48 images
        # 1. Modify first convolutional layer: smaller kernel and stride for GTSRB
        resnet18_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 2. Remove first max pooling layer to preserve spatial resolution
        resnet18_model.maxpool = nn.Identity()

        # 3. Modify classifier for 43 classes (GTSRB has 43 traffic sign classes)
        num_ftrs = resnet18_model.fc.in_features
        resnet18_model.fc = nn.Linear(num_ftrs, 43)

        self.model = resnet18_model.to(device)

        # Collect all BatchNorm layers for framework compatibility
        self.bn_layers = [
            module
            for module in self.model.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            - x: (torch.tensor) input tensor, must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 43)
        """
        return self.model(x)

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalized predictions
            - y:      (torch.tensor) true labels

        Returns:
            torch.tensor containing scalar accuracy value
        """
        _, preds = torch.max(logits, dim=1)
        correct = (preds == y).sum().float()
        acc = correct / y.size(0)
        return acc

    def empty_step(self):
        """
        Perform one step of SGD with dummy data to initialize optimizer parameters.
        """
        # Create dummy input and target for GTSRB (48x48 images)
        dummy_input = torch.zeros((2, 3, 48, 48), device=self.device)
        dummy_target = torch.zeros((2), dtype=torch.long, device=self.device)

        # Perform one training step
        self.train_step(dummy_input, dummy_target)


class UrbanSound8KModel(FLModel):
    """
    SOTA Audio-CNN model for UrbanSound8K dataset (mel-spectrograms, 10 classes).
    Designed specifically for mel-spectrogram classification with:
    - 4 convolutional blocks with increasing filters
    - Batch normalization for stable training
    - Max pooling for dimension reduction
    - Dropout for regularization
    """

    def __init__(self, device, n_mels=128, n_frames=173):
        """
        Initialize UrbanSound8KModel with Audio-CNN architecture.

        Args:
            - device: (torch.device) device to place model on
            - n_mels: (int) number of mel frequency bands (height of spectrogram)
            - n_frames: (int) number of time frames (width of spectrogram)
        """
        super(UrbanSound8KModel, self).__init__(device)

        # Store dimensions for empty_step
        self.n_mels = n_mels
        self.n_frames = n_frames

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        ).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.bn1 = nn.BatchNorm2d(16).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=2).to(device)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.bn2 = nn.BatchNorm2d(32).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=2).to(device)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1).to(device)
        self.relu3 = nn.ReLU().to(device)
        self.bn3 = nn.BatchNorm2d(64).to(device)
        self.pool3 = nn.MaxPool2d(kernel_size=2).to(device)

        # Convolutional Block 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
        self.relu4 = nn.ReLU().to(device)
        self.bn4 = nn.BatchNorm2d(128).to(device)
        self.pool4 = nn.MaxPool2d(kernel_size=2).to(device)

        # Flatten layer
        self.flatten = nn.Flatten().to(device)

        # Calculate the size after all pooling operations
        # After 4 max pools with kernel_size=2: dimension reduces by 2^4 = 16
        h_out = n_mels // 16
        w_out = n_frames // 16

        # Classifier head
        self.fc1 = nn.Linear(128 * h_out * w_out, 128).to(device)
        self.relu5 = nn.ReLU().to(device)
        self.dropout = nn.Dropout(0.5).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)  # 10 classes for UrbanSound8K

        # Collect all BatchNorm layers for framework compatibility
        self.bn_layers = [self.bn1, self.bn2, self.bn3, self.bn4]

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize weights for better training stability
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights using Xavier/Glorot initialization for better training stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier initialization for convolutional layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard initialization for batch normalization
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through the Audio-CNN.

        Args:
            - x: (torch.tensor) input mel-spectrogram, shape (batch_size, 1, n_mels, n_frames)

        Returns:
            torch.tensor model outputs (logits), shape (batch_size, 10)
        """
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.pool4(x)

        # Flatten and classify
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalized predictions
            - y:      (torch.tensor) true labels

        Returns:
            torch.tensor containing scalar accuracy value
        """
        _, preds = torch.max(logits, dim=1)
        correct = (preds == y).sum().float()
        acc = correct / y.size(0)
        return acc

    def empty_step(self):
        """
        Perform one step of SGD with dummy data to initialize optimizer parameters.
        """
        # Create dummy input and target for UrbanSound8K
        # Shape: (batch_size=2, channels=1, n_mels, n_frames)
        dummy_input = torch.zeros(
            (2, 1, self.n_mels, self.n_frames), device=self.device
        )
        dummy_target = torch.zeros((2), dtype=torch.long, device=self.device)

        # Perform one training step
        self.train_step(dummy_input, dummy_target)


class MINDModel(FLModel):
    """
    Code Reference for Text-CNN Structure: Kim Y. Convolutional neural networks for sentence classification[J]. arXiv preprint arXiv:1408.5882, 2014.
    Text-CNN model for MIND news classification dataset.
    Implements a classic CNN architecture for text classification with:
    - Pre-trained GloVe embeddings
    - Multiple convolutional filters with different kernel sizes
    - Max-over-time pooling
    - Dropout regularization
    """

    def __init__(
        self,
        device,
        embedding_matrix,
        num_classes,
        dropout_rate=0.5,
        freeze_embeddings=True,
        max_seq_len=30,
    ):
        """
        Initialize MINDModel with Text-CNN architecture.

        Args:
            - device: (torch.device) device to place model on
            - embedding_matrix: (np.ndarray) pre-trained embedding matrix from GloVe
            - num_classes: (int) number of output classes
            - dropout_rate: (float) dropout probability
            - freeze_embeddings: (bool) whether to freeze embedding layer weights
            - max_seq_len: (int) maximum sequence length for input
        """
        super(MINDModel, self).__init__(device)

        # Store max_seq_len for empty_step
        self.max_seq_len = max_seq_len

        # Embedding layer with pre-trained weights
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings,
            padding_idx=0,  # PAD token index
        ).to(device)

        # CNN parameters (SOTA configuration)
        num_filters = 100
        kernel_sizes = [3, 4, 5]  # Classic Text-CNN kernel sizes

        # Convolutional layers
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim, out_channels=num_filters, kernel_size=k
                ).to(device)
                for k in kernel_sizes
            ]
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate).to(device)

        # Fully connected layer
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes).to(device)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # No batch normalization layers for Text-CNN
        self.bn_layers = []

    def forward(self, x):
        """
        Forward pass through Text-CNN.

        Args:
            - x: (torch.tensor) input tensor of token indices, shape (batch_size, max_seq_len)

        Returns:
            torch.tensor model outputs (logits), shape (batch_size, num_classes)
        """
        # Embedding lookup: (batch_size, max_seq_len) -> (batch_size, max_seq_len, embedding_dim)
        x = self.embedding(x)

        # Transpose for Conv1d: (batch_size, max_seq_len, embedding_dim) -> (batch_size, embedding_dim, max_seq_len)
        x = x.permute(0, 2, 1)

        # Apply convolutions with different kernel sizes
        conv_outputs = []
        for conv in self.convs:
            # Conv1d: (batch_size, embedding_dim, max_seq_len) -> (batch_size, num_filters, conv_seq_len)
            conv_out = F.relu(conv(x))

            # Max-over-time pooling: (batch_size, num_filters, conv_seq_len) -> (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            pooled = pooled.squeeze(2)  # Remove last dimension
            conv_outputs.append(pooled)

        # Concatenate all pooled outputs: (batch_size, len(kernel_sizes) * num_filters)
        concatenated = torch.cat(conv_outputs, dim=1)

        # Apply dropout
        dropped = self.dropout(concatenated)

        # Final classification layer
        logits = self.fc(dropped)

        return logits

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalized predictions
            - y:      (torch.tensor) true labels

        Returns:
            torch.tensor containing scalar accuracy value
        """
        _, preds = torch.max(logits, dim=1)
        correct = (preds == y).sum().float()
        acc = correct / y.size(0)
        return acc

    def empty_step(self):
        """
        Perform one step of SGD with dummy data to initialize optimizer parameters.
        """
        # Create dummy input and target for MIND dataset
        # Shape: (batch_size=2, max_seq_len)
        dummy_input = torch.zeros(
            (2, self.max_seq_len), dtype=torch.long, device=self.device
        )
        dummy_target = torch.zeros((2), dtype=torch.long, device=self.device)

        # Perform one training step
        self.train_step(dummy_input, dummy_target)


class NumpyModel:
    """
    Allows easy operations on whole model of parameters using numpy arrays.
    """

    def __init__(self, params):
        """
        Initialize with a list or NumPy array of parameters.

        Args:
            - params: list of numpy arrays representing model parameters.
        """
        self.params = params

    def copy(self):
        """
        Returns:
            (NumpyModel) with all parameters copied from this model.
        """
        return NumpyModel([np.copy(p) for p in self.params])

    def zeros_like(self):
        """
        Returns:
            (NumpyModel) with all-0 values.
        """
        return NumpyModel([np.zeros_like(p) for p in self.params])

    def _op(self, other, f):
        """
        Return a new NumpyModel, where each parameter is computed using function
        f of this model's parameters and the other model's corresponding
        parameters/a constant value.

        Args:
            - other: (NumpyModel) or float/int
            - f:     (function)   to apply
        """
        if np.isscalar(other):
            new_params = [f(p, other) for p in self.params]

        elif isinstance(other, NumpyModel):
            new_params = [f(p, o) for (p, o) in zip(self.params, other.params)]

        else:
            raise ValueError("Incompatible type for op: {}".format(other))

        return NumpyModel(new_params)

        #     # Ensure parameters are at least 1-dimensional before concatenation [Must pay attention, important!!]

        #                   zip(self.params, other.params)]
        # else:
        #     raise ValueError(f'Incompatible type for op: {type(other)}')

    def abs(self):
        """
        Returns:
            (NumpyModel) with all absolute values.
        """
        return NumpyModel([np.absolute(p) for p in self.params])

    def __add__(self, other):
        """
        Args:
            - other: (NumpyModel) or float/int.

        Returns:
            (NumpyModel) of self + other, elementwise.
        """
        return self._op(other, np.add)

    def __radd__(self, other):
        """
        Returns new NumpyModel with vals of (self + other).

        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, np.add)

    def __sub__(self, other):
        """
        Returns new NumpyModel with vals of (self - other).

        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.sub)

    def __mul__(self, other):
        """
        Returns new NumpyModel with vals of (self * other).

        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        """
        Returns new NumpyModel with vals of (self * other).

        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.mul)

    def __truediv__(self, other):
        """
        Returns new NumpyModel with vals of (self / other).

        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.truediv)

    def __pow__(self, other):
        """
        Returns new NumpyModel with vals of (self ^ other).

        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.pow)

    def __getitem__(self, key):
        """
        Get parameter [key] of model.
        """
        return self.params[key]

    def __len__(self):
        """
        Return number of parameters in model.
        """
        return len(self.params)

    def __iter__(self):
        """
        Iterate over parameters of model.
        """
        for p in self.params:
            yield p
