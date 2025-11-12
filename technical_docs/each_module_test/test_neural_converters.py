# test_neural_converters.py
import numpy as np
from utils.neural_converters import NeuralConverter, SpikingConverter


def test_neural_converters():
    print("=== Neural Converter Test ===\n")

    # Initialize
    neural_conv = NeuralConverter()
    spike_conv = SpikingConverter()

    # Test 1: Rate to spike conversion
    print("1. Rate to spike conversion:")

    rates = np.array([10, 50, 100, 20, 0])  # Hz
    duration = 100  # ms

    spike_trains = neural_conv.rate_to_spike(rates, duration)

    print(f"Input rates (Hz): {rates}")
    print(f"Spike train shape: {spike_trains.shape}")

    # Count spikes for each neuron
    spike_counts = np.sum(spike_trains, axis=1)
    print(f"Spike counts: {spike_counts}")
    print(f"Expected counts: {rates * duration / 1000}")

    # Test 2: Spike to rate conversion
    print("\n2. Spike to rate conversion:")

    recovered_rates = neural_conv.spike_to_rate(spike_trains, window=50)
    print(f"Recovered rates: {recovered_rates}")
    print(f"Original rates: {rates}")
    print(f"Error: {np.abs(recovered_rates - rates)}")

    # Test 3: Vector to population code
    print("\n3. Vector to population code:")

    vector = np.array([0.5, -0.3, 0.8])
    population = neural_conv.vector_to_population(vector, n_neurons_per_dim=10)

    print(f"Input vector: {vector}")
    print(f"Population code shape: {population.shape}")
    print(f"Population activity: {population[:10]}")  # First dimension

    # Test 4: Population to vector decoding
    print("\n4. Population to vector decoding:")

    decoded = neural_conv.population_to_vector(population, n_neurons_per_dim=10)
    print(f"Decoded vector: {decoded}")
    print(f"Original vector: {vector}")
    print(f"Decoding error: {np.abs(decoded - vector)}")

    # Test 5: Spike patterns
    print("\n5. Creating spike patterns:")

    patterns = ["synchronous", "wave", "random", "burst"]

    for pattern in patterns:
        spikes = spike_conv.create_spike_pattern(pattern, n_neurons=20, duration=200)
        total_spikes = np.sum(spikes)
        print(f"\n{pattern.capitalize()} pattern:")
        print(f"  Total spikes: {total_spikes}")
        print(f"  Spikes per neuron: {np.sum(spikes, axis=1)[:5]}...")

    # Test 6: Temporal coding
    print("\n6. Temporal coding:")

    values = [0.1, 0.5, 0.9]

    for value in values:
        spike_train = spike_conv.encode_value_temporal(value, duration=100)
        decoded_value = spike_conv.decode_temporal(spike_train)

        spike_time = np.where(spike_train > 0)[0]
        print(f"\nValue: {value}")
        print(f"  Spike time: {spike_time[0] if len(spike_time) > 0 else 'No spike'}")
        print(f"  Decoded: {decoded_value:.2f}")

    # Test 7: Spike train distance
    print("\n7. Spike train distance:")

    train1 = spike_conv.create_spike_pattern("random", n_neurons=1, duration=100)[0]
    train2 = spike_conv.create_spike_pattern("random", n_neurons=1, duration=100)[0]

    distance = spike_conv.spike_distance(train1, train2, metric="victor_purpura")

    print(f"Train 1 spikes: {np.sum(train1)}")
    print(f"Train 2 spikes: {np.sum(train2)}")
    print(f"Victor-Purpura distance: {distance:.2f}")


test_neural_converters()