import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(data: np.ndarray):
    """Fits the scaler on ALL of the data passed in.

    Warning: if the full dataset is passed here before splitting, the scaler sees val
    and test. That is leakage, and holdout metrics become more optimistic than they
    really are. Project rule: the scaler must be fitted on the training split only.
    For training, use scale_data_train_only().

    The function is kept as is: it is correct when only the training part is
    deliberately passed in.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler


def train_rows_upper_bound(total_rows, window_size, steps_ahead, val_size, test_size):
    """How many leading rows belong to the training sequences.

    split_data splits sequences, not rows, so the row boundary is computed
    separately. The last training sequence with index s uses rows
    [s, s + window_size + steps_ahead), so the upper bound is the index of the
    first validation sequence plus the window and horizon length.
    """
    total_sequences = total_rows - window_size - steps_ahead + 1
    if total_sequences <= 0:
        return 0

    test_start = int(total_sequences * (1 - test_size))
    val_ratio = val_size / (1 - test_size) if (1 - test_size) != 0 else 0
    val_start = int(test_start * (1 - val_ratio))
    if val_start <= 0:
        return 0
    return min(total_rows, val_start + window_size + steps_ahead - 1)


def scale_data_train_only(data: np.ndarray, window_size, steps_ahead, val_size, test_size):
    """Fits the scaler on training rows only and applies it to the whole array.

    This keeps the holdout truly unseen by preprocessing. A side effect worth
    remembering: val and test values may fall outside [0, 1] if the market moved
    beyond the training range. This is not a bug but an honest reflection of what
    the model will face in production.
    """
    train_rows = train_rows_upper_bound(len(data), window_size, steps_ahead, val_size, test_size)
    if train_rows <= 0:
        raise ValueError(
            f"Not enough data for the training part: rows {len(data)}, "
            f"window {window_size}, horizon {steps_ahead}"
        )
    scaler = MinMaxScaler()
    scaler.fit(data[:train_rows])
    return scaler.transform(data), scaler, train_rows

def augment_data(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise

def create_sequences(data, window_size, steps_ahead, target_col_idx=0):
    X, y = [], []
    for i in range(len(data) - window_size - steps_ahead + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size : i + window_size + steps_ahead, target_col_idx])
    return np.array(X), np.array(y)

def train_val_test_split(X, y, val_size=0.15, test_size=0.15):
    total = len(X)
    test_start = int(total * (1 - test_size))
    X_temp, X_test = X[:test_start], X[test_start:]
    y_temp, y_test = y[:test_start], y[test_start:]
    val_ratio = val_size / (1 - test_size)
    val_start = int(len(X_temp) * (1 - val_ratio))
    X_train, X_val = X_temp[:val_start], X_temp[val_start:]
    y_train, y_val = y_temp[:val_start], y_temp[val_start:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def split_data(X: np.ndarray, y: np.ndarray, val_size: float = 0.15, test_size: float = 0.15):
    total_samples = len(X)
    test_start = int(total_samples * (1 - test_size))
    X_temp, X_test = X[:test_start], X[test_start:]
    y_temp, y_test = y[:test_start], y[test_start:]
    val_ratio = val_size / (1 - test_size) if (1 - test_size) != 0 else 0
    val_start = int(len(X_temp) * (1 - val_ratio))
    X_train, X_val = X_temp[:val_start], X_temp[val_start:]
    y_train, y_val = y_temp[:val_start], y_temp[val_start:]
    return X_train, X_val, X_test, y_train, y_val, y_test
