# Load libraries
import numpy as np
import onnx
import onnxruntime  as ort
from vnnlib.compat import read_vnnlib_simple

# Network parameters
N = 14
L = 11

# Load the ONNX model into memory
model_path = "14_ieee/ldf14bus.onnx"
model = onnx.load(model_path)
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess = ort.InferenceSession(model_path, sess_options=sess_options)

# Get information about the input and output nodes of the ONNX model
input_info = sess.get_inputs()
output_info = sess.get_outputs()
input_name = input_info[0].name
input_shape = input_info[0].shape
output_name = output_info[0].name
output_shape = output_info[0].shape

# Load the vnnlib file into memory
vnnlib_path = "14_ieee/14_bus_prop1.vnnlib"
vnnlib = read_vnnlib_simple(vnnlib_path, input_shape[1], output_shape[1])
vnnlib_input = vnnlib[0][0]
vnnlib_output = vnnlib[0][1]

# create input data
input_val = [vnnlib_input[i][0] for i in range(2*L)]
input_data = np.array([input_val], dtype=np.float32)

# get the model output
output = sess.run([output_name], {input_name: input_data})

# extract power balance violation
result = np.dot(vnnlib_output[0][0], output[0][0])