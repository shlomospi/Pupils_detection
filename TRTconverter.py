from tensorflow.python.compiler.tensorrt import trt_convert as trt
from prep_data import *
import tensorflow as tf


def batch_input(batch_size=64):
    images, _ = prep_data(data=["RGB"],
                          binary=False,
                          res=config["res"],
                          thresh=config["thresholds"]["0"],
                          verbose = 2,
                          normalize=False)

    batched_input = images[:batch_size, :, :, :]

    batched_input = tf.constant(batched_input)
    return batched_input


def convert_to_trt_graph_and_save(precision_mode='float32',
                                  input_saved_model_dir='saved_model'):
    """
    Loads a keras model, saves a TF model, converts to a TRT engine and saves it.
    :param precision_mode: float16 or float 32
    :param input_saved_model_dir: relative path to folder which contains
                                  the keras model to be converted
    :return:
    """
    print("loading keras model..")
    model = tf.keras.models.load_model(input_saved_model_dir)

    print("converting to TF saved model..")
    TF_input_saved_model_dir = 'TF_'+input_saved_model_dir
    model.save(TF_input_saved_model_dir)
    print("Done")
    if precision_mode == 'float32':
        precision_mode = trt.TrtPrecisionMode.FP32
        converted_save_suffix = '_TFTRT_FP32'

    elif precision_mode == 'float16':
        precision_mode = trt.TrtPrecisionMode.FP16
        converted_save_suffix = '_TFTRT_FP16'

    # elif precision_mode == 'int8': # TODO int8
    #     precision_mode = trt.TrtPrecisionMode.INT8
    #     converted_save_suffix = '_TFTRT_INT8'
    #     raise SystemExit("Supported precisions: float32 float16")

    else:
        raise SystemExit("Supported precisions: float32 float16")
    output_saved_model_dir = TF_input_saved_model_dir + converted_save_suffix

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision_mode,
                                                                   max_workspace_size_bytes=8000000000,
                                                                   minimum_segment_size=1)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=TF_input_saved_model_dir,
                                        conversion_params=conversion_params)

    print('Converting {} to TF-TRT graph precision mode {}...'.format(TF_input_saved_model_dir, precision_mode))

    # if precision_mode == trt.TrtPrecisionMode.INT8:
    #
    #     # Here we define a simple generator to yield calibration data
    #     def calibration_input_fn():
    #         yield (calibration_data,)
    #
    #     # When performing INT8 optimization, we must pass a calibration function to convert
    #     converter.convert(calibration_input_fn=calibration_input_fn)
    #
    # else:
    converter.convert()
    converter.build(input_fn=input_fn)
    print('Saving converted model to {}...'.format(output_saved_model_dir))
    converter.save(output_saved_model_dir=output_saved_model_dir)
    print('Complete')


def input_fn():
    """
    Helper function that yields the shape and type of inputs
    that should be accepted by the optimized engine
    :return:
    """
    Inp1 = np.random.normal(size=(1, 36, 64, 1)).astype(np.uint8)
    yield (Inp1, )


def load_tf_saved_model(input_saved_model_dir):

    print('Loading saved model {}...'.format(input_saved_model_dir))
    saved_model_loaded = tf.saved_model.load(input_saved_model_dir, tags=[tf.python.saved_model.tag_constants.SERVING])

    infer = saved_model_loaded.signatures['serving_default']
    return infer


if __name__ == '__main__':
    TRT_parser = argparse.ArgumentParser(description="perp data")
    TRT_parser.add_argument("-p", '--precision',
                            default="16",
                            help='16 for float16, 32 for float 32')
    TRT_args = TRT_parser.parse_args()
    if TRT_args.precision == "16":
        precision = "float16"
    elif TRT_args.precision == "32":
        precision = "float32"
    else:
        raise SystemExit("only Float16 and Float 32 are supported")
    convert_to_trt_graph_and_save(precision_mode=precision)
