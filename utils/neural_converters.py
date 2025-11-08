import numpy as np


class NeuralConverter:
    """Convert between different neural representations"""

    @staticmethod
    def rate_to_spike(rates: np.ndarray,
                      duration: int = 100,
                      dt: float = 1.0) -> np.ndarray:
        """Convert rate-coded neurons to spike trains"""

        n_neurons = len(rates)
        spike_trains = np.zeros((n_neurons, duration))

        for i, rate in enumerate(rates):
            # Poisson spike generation
            prob = rate * dt / 1000.0  # Convert Hz to probability
            spikes = np.random.random(duration) < prob
            spike_trains[i] = spikes.astype(float)

        return spike_trains

    @staticmethod
    def spike_to_rate(spike_trains: np.ndarray,
                      window: int = 50) -> np.ndarray:
        """Convert spike trains to firing rates"""

        n_neurons, duration = spike_trains.shape
        rates = np.zeros(n_neurons)

        for i in range(n_neurons):
            # Count spikes in window
            spike_count = np.sum(spike_trains[i, -window:])
            rates[i] = spike_count / (window / 1000.0)  # Convert to Hz

        return rates

    @staticmethod
    def vector_to_population(vector: np.ndarray,
                             n_neurons_per_dim: int = 10) -> np.ndarray:
        """Convert vector to population code"""

        n_dims = len(vector)
        n_neurons = n_dims * n_neurons_per_dim
        population = np.zeros(n_neurons)

        for i, value in enumerate(vector):
            # Create tuning curves for this dimension
            start_idx = i * n_neurons_per_dim

            # Gaussian tuning curves
            centers = np.linspace(-1, 1, n_neurons_per_dim)
            width = 2.0 / n_neurons_per_dim

            for j, center in enumerate(centers):
                response = np.exp(-(value - center) ** 2 / (2 * width ** 2))
                population[start_idx + j] = response

        return population

    @staticmethod
    def population_to_vector(population: np.ndarray,
                             n_neurons_per_dim: int = 10) -> np.ndarray:
        """Decode vector from population code"""

        n_neurons = len(population)
        n_dims = n_neurons // n_neurons_per_dim
        vector = np.zeros(n_dims)

        for i in range(n_dims):
            start_idx = i * n_neurons_per_dim
            dim_population = population[start_idx:start_idx + n_neurons_per_dim]

            # Decode using center of mass
            centers = np.linspace(-1, 1, n_neurons_per_dim)

            if np.sum(dim_population) > 0:
                vector[i] = np.sum(centers * dim_population) / np.sum(dim_population)

        return vector


class SpikingConverter:
    """Utilities for spiking neural networks"""

    @staticmethod
    def create_spike_pattern(pattern: str,
                             n_neurons: int = 100,
                             duration: int = 1000) -> np.ndarray:
        """Create specific spike patterns"""

        spike_trains = np.zeros((n_neurons, duration))

        if pattern == "synchronous":
            # All neurons spike together
            spike_times = np.arange(0, duration, 50)
            for t in spike_times:
                if t < duration:
                    spike_trains[:, t] = 1

        elif pattern == "wave":
            # Traveling wave
            for i in range(n_neurons):
                delay = i * 5
                spike_times = np.arange(delay, duration, 100)
                for t in spike_times:
                    if t < duration:
                        spike_trains[i, t] = 1

        elif pattern == "random":
            # Random Poisson spikes
            rate = 20  # Hz
            prob = rate / 1000.0
            spike_trains = (np.random.random((n_neurons, duration)) < prob).astype(float)

        elif pattern == "burst":
            # Burst firing
            burst_times = np.arange(0, duration, 200)
            for burst_start in burst_times:
                burst_duration = 50
                for i in range(n_neurons):
                    burst_spikes = np.random.random(burst_duration) < 0.5
                    end = min(burst_start + burst_duration, duration)
                    spike_trains[i, burst_start:end] = burst_spikes[:end - burst_start]

        return spike_trains

    @staticmethod
    def encode_value_temporal(value: float,
                              duration: int = 100,
                              max_delay: int = 50) -> np.ndarray:
        """Encode value as spike timing (temporal coding)"""

        spike_train = np.zeros(duration)

        # Map value to delay (earlier spike = higher value)
        delay = int((1 - value) * max_delay)

        if delay < duration:
            spike_train[delay] = 1

        return spike_train

    @staticmethod
    def decode_temporal(spike_train: np.ndarray,
                        max_delay: int = 50) -> float:
        """Decode value from spike timing"""

        # Find first spike
        spike_indices = np.where(spike_train > 0)[0]

        if len(spike_indices) > 0:
            first_spike = spike_indices[0]
            value = 1 - (first_spike / max_delay)
            return np.clip(value, 0, 1)

        return 0.0

    @staticmethod
    def spike_distance(train1: np.ndarray,
                       train2: np.ndarray,
                       metric: str = "victor_purpura") -> float:
        """Calculate distance between spike trains"""

        if metric == "victor_purpura":
            # Victor-Purpura distance with cost parameter
            cost = 1.0

            # Get spike times
            spikes1 = np.where(train1 > 0)[0]
            spikes2 = np.where(train2 > 0)[0]

            if len(spikes1) == 0 and len(spikes2) == 0:
                return 0.0

            if len(spikes1) == 0:
                return len(spikes2) * cost

            if len(spikes2) == 0:
                return len(spikes1) * cost

            # Dynamic programming for VP distance
            n1, n2 = len(spikes1), len(spikes2)
            dp = np.zeros((n1 + 1, n2 + 1))

            # Initialize
            for i in range(1, n1 + 1):
                dp[i, 0] = i * cost
            for j in range(1, n2 + 1):
                dp[0, j] = j * cost

            # Fill matrix
            for i in range(1, n1 + 1):
                for j in range(1, n2 + 1):
                    time_diff = abs(spikes1[i - 1] - spikes2[j - 1])
                    dp[i, j] = min(
                        dp[i - 1, j] + cost,  # Delete
                        dp[i, j - 1] + cost,  # Insert
                        dp[i - 1, j - 1] + time_diff  # Shift
                    )

            return dp[n1, n2]

        elif metric == "isi":
            # Inter-spike interval distance
            isi1 = np.diff(np.where(train1 > 0)[0])
            isi2 = np.diff(np.where(train2 > 0)[0])

            if len(isi1) == 0 and len(isi2) == 0:
                return 0.0

            # Pad to same length
            max_len = max(len(isi1), len(isi2))
            isi1_pad = np.pad(isi1, (0, max_len - len(isi1)), constant_values=0)
            isi2_pad = np.pad(isi2, (0, max_len - len(isi2)), constant_values=0)

            return np.linalg.norm(isi1_pad - isi2_pad)

        else:
            # Simple spike count difference
            return abs(np.sum(train1) - np.sum(train2))