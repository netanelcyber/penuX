import math


def softmax_row(logits_row):
    """
    Stable softmax for one row of logits.
    """
    if not logits_row:
        return []
    m = max(logits_row)
    exps = [math.exp(x - m) for x in logits_row]
    s = sum(exps)
    if s == 0.0:
        return [0.0 for _ in logits_row]
    return [x / s for x in exps]


def argmax(values):
    """
    Pure Python argmax.
    """
    if not values:
        raise ValueError("argmax() received an empty list")
    best_idx = 0
    best_val = values[0]
    for i in range(1, len(values)):
        if values[i] > best_val:
            best_val = values[i]
            best_idx = i
    return best_idx


class PurePythonECE:
    """
    Expected Calibration Error in pure Python.

    Partitions confidence into equally spaced bins on [0, 1],
    then computes the weighted average of |accuracy - confidence|
    across non-empty bins.
    """
    def __init__(self, n_bins=15):
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        self.n_bins = int(n_bins)
        self.bin_lowers = [i / self.n_bins for i in range(self.n_bins)]
        self.bin_uppers = [(i + 1) / self.n_bins for i in range(self.n_bins)]

    def __call__(self, logits, labels):
        """
        Args:
            logits: list of rows, shape [N, C]
            labels: list of ints, shape [N]

        Returns:
            float ECE
        """
        if len(logits) != len(labels):
            raise ValueError("logits and labels must have the same length")
        if len(logits) == 0:
            return 0.0

        confidences = []
        predictions = []
        accuracies = []

        for row, label in zip(logits, labels):
            probs = softmax_row(row)
            pred = argmax(probs)
            conf = probs[pred]

            confidences.append(conf)
            predictions.append(pred)
            accuracies.append(1.0 if pred == label else 0.0)

        n = float(len(labels))
        ece = 0.0

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin_idx = []
            for i, conf in enumerate(confidences):
                if conf > bin_lower and conf <= bin_upper:
                    in_bin_idx.append(i)

            if not in_bin_idx:
                continue

            prop_in_bin = len(in_bin_idx) / n
            acc_sum = 0.0
            conf_sum = 0.0

            for i in in_bin_idx:
                acc_sum += accuracies[i]
                conf_sum += confidences[i]

            accuracy_in_bin = acc_sum / len(in_bin_idx)
            avg_confidence_in_bin = conf_sum / len(in_bin_idx)

            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class TemperatureScaler:
    """
    Pure Python post-hoc temperature scaling.

    Learns a single positive scalar T such that:
        scaled_logits = logits / T

    Optimization is done with gradient descent on mean cross-entropy.
    """
    def __init__(self, init_temperature=1.5):
        if init_temperature <= 0:
            raise ValueError("init_temperature must be > 0")
        self.temperature = float(init_temperature)

    def scale_row(self, logits_row):
        t = max(self.temperature, 1e-8)
        return [x / t for x in logits_row]

    def scale_logits(self, logits):
        return [self.scale_row(row) for row in logits]

    def _mean_cross_entropy(self, logits, labels):
        if len(logits) != len(labels):
            raise ValueError("logits and labels must have the same length")
        if len(logits) == 0:
            return 0.0

        total = 0.0
        eps = 1e-12

        for row, y in zip(logits, labels):
            scaled = self.scale_row(row)
            probs = softmax_row(scaled)
            py = probs[y]
            if py < eps:
                py = eps
            total += -math.log(py)

        return total / len(labels)

    def _temperature_gradient(self, logits, labels):
        """
        Gradient of mean CE wrt temperature T.

        For one sample:
            z_i = l_i / T
            L = CE(z, y)

        dL/dT = (l_y - sum_i p_i * l_i) / T^2

        Returns:
            mean gradient over samples
        """
        if len(logits) != len(labels):
            raise ValueError("logits and labels must have the same length")
        if len(logits) == 0:
            return 0.0

        t = max(self.temperature, 1e-8)
        grad_sum = 0.0

        for row, y in zip(logits, labels):
            scaled = [x / t for x in row]
            probs = softmax_row(scaled)

            expected_logit = 0.0
            for p, l in zip(probs, row):
                expected_logit += p * l

            ly = row[y]
            grad = (ly - expected_logit) / (t * t)
            grad_sum += grad

        return grad_sum / len(labels)

    def fit(self, valid_logits, valid_labels, lr=0.01, max_iter=200, tol=1e-7, verbose=False):
        """
        Fits temperature using gradient descent on mean cross-entropy.

        Args:
            valid_logits: list of rows [N, C]
            valid_labels: list of ints [N]
            lr: learning rate
            max_iter: number of optimization steps
            tol: stop if update magnitude is smaller than tol
            verbose: print optimization progress

        Returns:
            learned temperature (float)
        """
        if len(valid_logits) != len(valid_labels):
            raise ValueError("valid_logits and valid_labels must have the same length")
        if len(valid_logits) == 0:
            return self.temperature

        prev_loss = self._mean_cross_entropy(valid_logits, valid_labels)

        for step in range(max_iter):
            grad = self._temperature_gradient(valid_logits, valid_labels)

            new_t = self.temperature - lr * grad
            if new_t <= 1e-6:
                new_t = 1e-6

            delta = abs(new_t - self.temperature)
            self.temperature = new_t

            cur_loss = self._mean_cross_entropy(valid_logits, valid_labels)

            if verbose:
                print(
                    f"[TempScale step={step+1:03d}] "
                    f"T={self.temperature:.6f} "
                    f"loss={cur_loss:.6f} "
                    f"grad={grad:.6f}"
                )

            if delta < tol:
                break

            if abs(prev_loss - cur_loss) < tol:
                break

            prev_loss = cur_loss

        return self.temperature


def accuracy_from_logits(logits, labels):
    """
    Pure Python accuracy from logits.
    """
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have the same length")
    if len(logits) == 0:
        return 0.0

    correct = 0
    for row, y in zip(logits, labels):
        probs = softmax_row(row)
        pred = argmax(probs)
        if pred == y:
            correct += 1
    return correct / len(labels)


def evaluate_logits(all_logits, all_labels, n_bins=15, temp_lr=0.01, temp_max_iter=200, verbose=False):
    """
    Pure Python evaluation directly from logits + labels.

    Args:
        all_logits: list of rows [N, C]
        all_labels: list of ints [N]

    Returns:
        accuracy, pre_ece, post_ece, optimal_temperature
    """
    if len(all_logits) != len(all_labels):
        raise ValueError("all_logits and all_labels must have the same length")
    if len(all_logits) == 0:
        raise ValueError("Empty evaluation set")

    accuracy = accuracy_from_logits(all_logits, all_labels)

    ece_metric = PurePythonECE(n_bins=n_bins)
    pre_ece = ece_metric(all_logits, all_labels)

    print(f"[*] Pre-Calibration Accuracy: {accuracy * 100:.2f}%")
    print(f"[*] Pre-Calibration ECE:      {pre_ece:.6f}")

    scaler = TemperatureScaler(init_temperature=1.5)
    optimal_t = scaler.fit(
        valid_logits=all_logits,
        valid_labels=all_labels,
        lr=temp_lr,
        max_iter=temp_max_iter,
        verbose=verbose,
    )

    scaled_logits = scaler.scale_logits(all_logits)
    post_ece = ece_metric(scaled_logits, all_labels)

    print(f"[*] Optimal Temperature:      {optimal_t:.6f}")
    print(f"[*] Post-Calibration ECE:     {post_ece:.6f}")

    return accuracy, pre_ece, post_ece, optimal_t


def evaluate_model(model, dataloader, n_bins=15, temp_lr=0.01, temp_max_iter=200, verbose=False):
    """
    Pure Python evaluation loop.

    Assumptions:
      - model is a regular Python callable:
            logits_batch = model(texts, numerics)
        where logits_batch is a list of rows, shape [B, C]

      - dataloader is any iterable that yields:
            texts, numerics, labels

        where:
            texts    = batch input
            numerics = batch input
            labels   = list of ints
    """
    all_logits = []
    all_labels = []

    for texts, numerics, labels in dataloader:
        logits_batch = model(texts, numerics)

        if len(logits_batch) != len(labels):
            raise ValueError("Model output batch size does not match labels batch size")

        for row in logits_batch:
            all_logits.append([float(x) for x in row])

        for y in labels:
            all_labels.append(int(y))

    return evaluate_logits(
        all_logits=all_logits,
        all_labels=all_labels,
        n_bins=n_bins,
        temp_lr=temp_lr,
        temp_max_iter=temp_max_iter,
        verbose=verbose,
    )


# =========================================================
# Example usage with precomputed logits
# =========================================================
if __name__ == "__main__":
    logits = [
        [2.4, 0.3, 0.1],
        [0.2, 1.8, 0.6],
        [0.1, 0.5, 2.0],
        [2.1, 0.2, 0.1],
        [0.3, 2.2, 0.4],
        [1.0, 0.9, 0.8],
    ]
    labels = [0, 1, 2, 0, 1, 2]

    accuracy, pre_ece, post_ece, temperature = evaluate_logits(
        all_logits=logits,
        all_labels=labels,
        n_bins=15,
        temp_lr=0.05,
        temp_max_iter=300,
        verbose=False,
    )
