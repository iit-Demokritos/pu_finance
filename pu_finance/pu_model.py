import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array
from pu_finance.pu_loss import PULoss
from sklearn.metrics import f1_score


class Denser(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.toarray()


class MLP_1(nn.Module):
    """Multi-layer Perceptron neural network implementation in PyTorch."""

    def __init__(
        self,
        input_dim: int,
    ):
        super(MLP_1, self).__init__()

        self.network = nn.Sequential(*[nn.Linear(input_dim, 1), nn.Sigmoid()])
        nn.init.xavier_uniform_(self.network[0].weight)

    def forward(self, x):
        return self.network(x)


class MLP(nn.Module):
    """Multi-layer Perceptron neural network implementation in PyTorch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout_rate: float = 0.2,
    ):
        super(MLP, self).__init__()

        # Build layers dynamically based on hidden_dims parameter
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PU_MLP(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for PyTorch MLP with PU Loss
    for binary classification with Positive-Unlabeled learning.
    """

    def __init__(
        self,
        hidden_dims: list[int] = [100, 50],
        dropout_rate: float = 0.2,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        patience: int = 10,
        device: str = "cuda:0",
        random_state: int = 42,
        prior: float = 0.5,
        nnPU: bool = False,
        verbose: bool = False,
    ):
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.random_state = random_state
        self.model = None
        self.prior = prior
        self.nnPU = nnPU
        self.verbose = verbose

    def _init_model(self, input_dim):
        """Initialize the PyTorch model with the correct input dimension."""
        torch.manual_seed(self.random_state) if self.random_state is not None else None
        np.random.seed(self.random_state) if self.random_state is not None else None

        self.model = MLP_1(input_dim=input_dim).to(self.device)
        # self.model = MLP(
        #     input_dim=input_dim,
        #     hidden_dims=self.hidden_dims,
        #     dropout_rate=self.dropout_rate,
        # ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = PULoss(prior=self.prior, nnPU=self.nnPU)  # nn.BCELoss()  #

    def fit(self, X, y):
        """
        Fit the MLP model using PU loss.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels (1 for positive, 0 for unlabeled).

        Returns:
        --------
        self : object
            Returns self.
        """
        # Check and validate input data
        X, y = check_X_y(X, y, accept_sparse=False)

        # Save dimensions
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.1,
            random_state=self.random_state,
            stratify=y,
        )

        # Initialize model
        self._init_model(X.shape[1])

        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)

        # Create DataLoader for training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Early stopping setup
        best_val_loss = float("inf")
        no_improve_epochs = 0
        best_model_state = None

        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            # Training
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                train_outputs = self.model(X_train_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                train_loss = self.criterion(train_outputs, y_train_tensor).item()
                f1 = f1_score(
                    y_train,
                    (train_outputs.detach().cpu().numpy() > 0.5).astype(int),
                    average="binary",
                )
                f1_val = f1_score(
                    y_val,
                    (val_outputs.detach().cpu().numpy() > 0.5).astype(int),
                    average="binary",
                )

            self.model.train()

            if self.verbose:
                if epoch % 5 == 0:
                    print(
                        f"Epoch {epoch + 1}\t Train loss: {epoch_loss:.4f} - {train_loss:.4f} (F1: {f1:.4f})\t Val loss: {val_loss:.4f} (F1: {f1_val:.4f})"
                    )
                    print(
                        f"{(val_outputs>0.5).sum().item()} / {y_val_tensor.sum().item()}"
                    )

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                best_model_state = self.model.state_dict().copy()
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.patience:
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data samples for prediction.

        Returns:
        --------
        array-like of shape (n_samples, 2)
            Returns probability estimates for both classes [P(y=0), P(y=1)].
        """
        # Check input
        X = check_array(X, accept_sparse=False)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            pos_proba = self.model(X_tensor).cpu().numpy().flatten()

        # Format as a 2D array with probabilities for both classes
        neg_proba = 1 - pos_proba
        return np.vstack((neg_proba, pos_proba)).T

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data samples for prediction.

        Returns:
        --------
        array-like of shape (n_samples,)
            Returns predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
