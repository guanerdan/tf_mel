import numpy as np
import tensorflow as tf

from tensorflow.python.framework import tensor_util

def _hann_window(frame_length, dtype=tf.float32):
    frame_length = 800
    fft_length = 1024
    
    window = tf.cast(
        tf.reshape(
            tf.constant(
                (
                    0.5
                    - 0.5
                    * np.cos(
                        2 * np.pi * np.arange(0, frame_length) / (frame_length - 1)
                    )
                ).astype(np.float32),
                name="hann_window",
            ),
            [1, frame_length],
        ),
        dtype=dtype,
    )

    frame_length_const = tensor_util.constant_value(frame_length)
    fft_length_const = tensor_util.constant_value(fft_length)
    
    if frame_length_const < fft_length_const:
        # https://github.com/pytorch/pytorch/issues/72328
        window = tf.reshape(window, [1, frame_length])
        pad_left = int((fft_length - frame_length) // 2)
        pad_right = int(fft_length - frame_length - pad_left)
        padding = tf.constant(
            [[0, 0], [pad_left, pad_right]], dtype=tf.dtypes.int32
        )
        window = tf.pad(window, padding, mode="CONSTANT")
        window = tf.reshape(window, [1, fft_length])

    return tf.reshape(window, [fft_length])


def tflite_stft_magnitude(signal, frame_length, frame_step, fft_length, center=False):
    """TF-Lite-compatible version of tf.abs(tf.signal.stft())."""

    def _dft_matrix(dft_length):
        """Calculate the full DFT matrix in NumPy."""
        # See https://en.wikipedia.org/wiki/DFT_matrix
        omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
        # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
        return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))

    def _rdft(framed_signal, fft_length):
        """Implement real-input Discrete Fourier Transform by matmul."""
        # We are right-multiplying by the DFT matrix, and we are keeping only the
        # first half ("positive frequencies").  So discard the second half of rows,
        # but transpose the array for right-multiplication.  The DFT matrix is
        # symmetric, so we could have done it more directly, but this reflects our
        # intention better.
        complex_dft_matrix_kept_values = _dft_matrix(fft_length)[
            : (fft_length // 2 + 1), :
        ].transpose()
        real_dft_matrix = tf.constant(
            np.real(complex_dft_matrix_kept_values).astype(np.float32),
            name="real_dft_matrix",
        )
        imag_dft_matrix = tf.constant(
            np.imag(complex_dft_matrix_kept_values).astype(np.float32),
            name="imaginary_dft_matrix",
        )
        signal_frame_length = tf.shape(framed_signal)[-1]
        half_pad = (fft_length - signal_frame_length) // 2
        padded_frames = tf.pad(
            framed_signal,
            [
                # Don't add any padding in the frame dimension.
                [0, 0],
                # Pad before and after the signal within each frame.
                [half_pad, fft_length - signal_frame_length - half_pad],
            ],
            mode="CONSTANT",
            constant_values=0.0,
        )
        real_stft = tf.matmul(padded_frames, real_dft_matrix)
        imag_stft = tf.matmul(padded_frames, imag_dft_matrix)
        return real_stft, imag_stft

    def _complex_abs(real, imag):
        return tf.sqrt(tf.add(real * real, imag * imag))

    window = tf.signal.hann_window(frame_length, periodic=False)
    
    if frame_length < fft_length:
        # https://github.com/pytorch/pytorch/issues/72328
        pad_left = (fft_length - frame_length) // 2
        pad_right = fft_length - frame_length - pad_left
        padding = tf.constant([[0, 0], [pad_left, pad_right]], dtype=tf.int32)
        window = tf.pad(window, padding, mode="CONSTANT")
        frame_length = fft_length
        print(window.shape)

    framed_signal = tf.signal.frame(signal, frame_length, frame_step, pad_end=False)
    windowed_signal = framed_signal * window
    real_stft, imag_stft = _rdft(windowed_signal, fft_length)
    stft_magnitude = _complex_abs(real_stft, imag_stft)
    return stft_magnitude
