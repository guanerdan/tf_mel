import os
import sys
import tensorflow as tf
import numpy as np
from functools import partial

file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, file_path)
from tflite_mel import tflite_stft_magnitude, _hann_window
from custom_mel_bank import custom_linear_to_mel_weight_matrix
from custom_stft import stft


def waveform_to_log_mel_spectrogram_patches(
    waveform,
    window_length_samples=800,
    hop_length_samples=320,
    conv_filter=np.array([[[-0.97]],[[1.]]]),
    stft_center=True,
    tflite_compatible=False,
    mel_bands=128,
    sample_rate=32000,
    mel_min_hz=0.0,
    mel_max_hz=15000,
    log_offset=0.00001
):
    """Compute log mel spectrogram patches of a 1-D waveform."""
    with tf.name_scope("log_mel_features"):
        # waveform has shape [<# samples>]

        # Convert waveform into spectrogram using a Short-Time Fourier Transform.
        # Note that tf.signal.stft() uses a periodic Hann window by default.

        window_length_samples = window_length_samples
        hop_length_samples = hop_length_samples

        fft_length = 2 ** int(
            np.ceil(np.log(window_length_samples) / np.log(2.0))
        )  # should be 1024
        num_spectrogram_bins = fft_length // 2 + 1

        if len(waveform.shape) < 3:
            waveform = tf.expand_dims(waveform, axis=2)

        conv_filter = tf.reshape(
            tf.constant(conv_filter, dtype=tf.float32), [2, 1, 1]
        )
        waveform = tf.nn.conv1d(
            input=waveform,
            filters=conv_filter,
            stride=1,
            padding="VALID",
            data_format="NWC",
        )
        waveform = tf.squeeze(waveform, axis=None)

        if stft_center:
            pad_samples = int(fft_length // 2)

            waveform = tf.expand_dims(waveform, axis=0)
            paddings = tf.constant([[0, 0], [pad_samples, pad_samples]])
            waveform = tf.pad(waveform, paddings, mode="REFLECT")
            waveform = tf.squeeze(waveform, axis=None)

        if tflite_compatible:
            magnitude_spectrogram = tflite_stft_magnitude(
                signal=waveform,
                frame_length=window_length_samples,
                frame_step=hop_length_samples,
                fft_length=fft_length,
            )
        else:
            magnitude_spectrogram = tf.abs(
                tf.signal.stft(
                    signals=waveform,
                    frame_length=fft_length,
                    frame_step=hop_length_samples,
                    fft_length=fft_length,
                    window_fn=_hann_window,
                )
            )

        # magnitude_spectrogram has shape [<# STFT frames>, num_spectrogram_bins]

        magnitude_spectrogram = tf.pow(magnitude_spectrogram, 2)  # power mag

        # Convert spectrogram into log mel spectrogram.
        linear_to_mel_weight_matrix = custom_linear_to_mel_weight_matrix(
            num_mel_bins=mel_bands,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=mel_min_hz,
            upper_edge_hertz=mel_max_hz,
        )

        mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)

        # mel log
        log_mel_spectrogram = tf.math.log(mel_spectrogram + log_offset)
        # log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands]

        # fast norm
        log_mel_spectrogram = (log_mel_spectrogram + 4.5) / 5.0

        return tf.transpose(log_mel_spectrogram)
    

if __name__ == '__main__':
    waveform = np.random.rand(1, 32000*10).astype(np.float32)
    outputs = waveform_to_log_mel_spectrogram_patches(waveform)
    print(outputs.shape)
