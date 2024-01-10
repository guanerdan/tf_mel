from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.ops.signal.mel_ops import _validate_arguments, _hertz_to_mel


def custom_linear_to_mel_weight_matrix(
    num_mel_bins=20,
    num_spectrogram_bins=129,
    sample_rate=8000,
    lower_edge_hertz=125.0,
    upper_edge_hertz=3800.0,
    dtype=dtypes.float32,
    name=None,
):
    with ops.name_scope(name, "linear_to_mel_weight_matrix") as name:
        # Convert Tensor `sample_rate` to float, if possible.
        if isinstance(sample_rate, tensor.Tensor):
            maybe_const_val = tensor_util.constant_value(sample_rate)
            if maybe_const_val is not None:
                sample_rate = maybe_const_val

        # Note: As num_spectrogram_bins is passed to `math_ops.linspace`
        # and the validation is already done in linspace (both in shape function
        # and in kernel), there is no need to validate num_spectrogram_bins here.
        _validate_arguments(
            num_mel_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, dtype
        )

        # This function can be constant folded by graph optimization since there are
        # no Tensor inputs.
        sample_rate = math_ops.cast(sample_rate, dtype, name="sample_rate")
        lower_edge_hertz = ops.convert_to_tensor(
            lower_edge_hertz, dtype, name="lower_edge_hertz"
        )
        upper_edge_hertz = ops.convert_to_tensor(
            upper_edge_hertz, dtype, name="upper_edge_hertz"
        )
        zero = ops.convert_to_tensor(0.0, dtype)

        # HTK excludes the spectrogram DC bin.
        # bands_to_zero = 1
        bands_to_zero = 0  # to complicant with torch.kaldi, we includes the DC bin
        nyquist_hertz = sample_rate / 2.0
        linear_frequencies = math_ops.linspace(
            zero, nyquist_hertz, num_spectrogram_bins
        )[bands_to_zero:]

        spectrogram_bins_mel = array_ops.expand_dims(
            _hertz_to_mel(linear_frequencies), 1
        )

        # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
        # center of each band is the lower and upper edge of the adjacent bands.
        # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
        # num_mel_bins + 2 pieces.
        band_edges_mel = shape_ops.frame(
            math_ops.linspace(
                _hertz_to_mel(lower_edge_hertz),
                _hertz_to_mel(upper_edge_hertz),
                num_mel_bins + 2,
            ),
            frame_length=3,
            frame_step=1,
        )

        # Split the triples up and reshape them into [1, num_mel_bins] tensors.
        lower_edge_mel, center_mel, upper_edge_mel = tuple(
            array_ops.reshape(t, [1, num_mel_bins])
            for t in array_ops.split(band_edges_mel, 3, axis=1)
        )

        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the mel domain, not Hertz.
        lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
            center_mel - lower_edge_mel
        )  # up_slope
        upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
            upper_edge_mel - center_mel
        )  # down_slope

        # Intersect the line segments with each other and zero.
        mel_weights_matrix = math_ops.maximum(
            zero, math_ops.minimum(lower_slopes, upper_slopes)
        )
        # print(mel_weights_matrix.numpy().T)

        # Re-add the zeroed lower bins we sliced out above.
        return array_ops.pad(
            mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], name=name
        )
