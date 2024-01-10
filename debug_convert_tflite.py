import tensorflow as tf
from tensorflow import keras
import onnx2tf
import numpy as np
import os

from tf_mel_lib.tf_mel import waveform_to_log_mel_spectrogram_patches


def load_tf_valid(path, input):
    print("Loading from", path)
    loaded = tf.saved_model.load(path)
    out = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY](
        tf.constant(input, dtype=tf.float32)
    )
    print("Output with input", input, ": ")
    for k, v in out.items():
        print(f"{k} : {v}")


def make_tfmodel_from_onnx(onnx_path, output_path):
    
    tf_func = onnx2tf.convert(
        onnx_path,
        output_folder_path=output_path,
        output_signaturedefs=True,
        # output_keras_v3=True,
        check_onnx_tf_outputs_elementwise_close_full=True,
        check_onnx_tf_outputs_elementwise_close_atol=1e-4,
        copy_onnx_input_output_names_to_tflite=True)

    dummy_input = np.random.rand(1, 128, 1000, 1)
    load_tf_valid(output_path, dummy_input)

    return tf_func


def make_pipeline_model(tf_model_path, output_path):
    
    tf_model = tf.saved_model.load(tf_model_path)
    f = tf_model.signatures["serving_default"]

    from tf_mel_lib.tf_mel import waveform_to_log_mel_spectrogram_patches
    
    class MN10_AS(tf.Module):
        @tf.function
        def inference(self, waveform):
            mel = waveform_to_log_mel_spectrogram_patches(waveform)
            mel = tf.reshape(mel, shape=(1, 128, 1000, 1))
            res = f(mel)
            score, feat = res['output_score'], res['output_feature']
            return score, feat
        
        @tf.function
        def signatures_func(self, waveform):
            score, feat = self.inference(waveform)
            return {"output_score": score, 'output_feature': feat}
        
    model = MN10_AS()
    res = model.inference(tf.random.uniform((1, 320000)))
    print(res)

    tf.saved_model.save(
        model, output_path, signatures=model.signatures_func.get_concrete_function(
            tf.TensorSpec([1, 320000], tf.float32)))

    dummy_input = np.random.rand(1, 320000)
    load_tf_valid(output_path, dummy_input)


def convert_tflite(saved_model_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()
    # interpreter = tf.lite.Interpreter(model_content=tflite_model)

    # Save the model.
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

def main():
    onnx_path = 'mn10_as_mobilenet.onnx'
    tf_model_output_path = 'mn10_as_mobilenet'
    pipelined_model_path = 'mn10_as'
    tflite_path = f"{pipelined_model_path}/mn10_as_float32.tflite"
    
    make_tfmodel_from_onnx(onnx_path, tf_model_output_path)
    make_pipeline_model(tf_model_output_path, pipelined_model_path)
    convert_tflite(pipelined_model_path, tflite_path)



if __name__ == '__main__':
    main()


    
