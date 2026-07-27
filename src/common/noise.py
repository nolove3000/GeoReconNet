import numpy as np


def add_measurement_noise(
    modal_raw,
    frequency_raw,
    mode_noise_percentage,
    frequency_noise_percentage,
    random_generator,
):
    """Apply multiplicative Gaussian noise to modal shapes and frequencies."""
    modal_noise = random_generator.normal(
        0.0, mode_noise_percentage, size=np.asarray(modal_raw).shape
    )
    frequency_noise = random_generator.normal(
        0.0, frequency_noise_percentage, size=np.asarray(frequency_raw).shape
    )
    return (
        np.asarray(modal_raw) * (1.0 + modal_noise),
        np.asarray(frequency_raw) * (1.0 + frequency_noise),
    )
