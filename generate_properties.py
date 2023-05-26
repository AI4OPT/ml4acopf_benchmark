import argparse
import numpy as np
import random
import onnx
import onnxruntime  as ort
import json
import os

def main(network_name, seed):
    random.seed(seed)  # set a specific seed value for reproducibility

    data_folder = "data"
    network_path = os.path.join(data_folder, f"{network_name}.ref.json")

    # Open the JSON file
    with open(network_path, 'r') as file:
        # Load the data from the file
        network = json.load(file)

    N = len(network['data']['bus'])
    L = len(network['data']['load'])

    model_path = os.path.join("onnx", f"{network_name}_ml4acopf.onnx")

    # Load the ONNX model into memory
    sess = ort.InferenceSession(model_path)

    # Get information about the input and output nodes of the ONNX model
    input_info = sess.get_inputs()
    output_info = sess.get_outputs()

    # Assume the first input and output nodes are the ones you want to use
    input_shape = input_info[0].shape
    output_shape = output_info[0].shape

    # Sort the keys of the load data dictionary
    sorted_load_keys = sorted(network['data']['load'].keys(), key=lambda x: int(x))
    sorted_bus_keys = sorted(network['data']['bus'].keys(), key=lambda x: int(x))

    bus_id_to_index = {}
    for idx, key in enumerate(sorted_bus_keys):
        bus_id = int(network['data']['bus'][key]['source_id'][1])
        bus_id_to_index[bus_id] = idx

    # Reference load
    pd = [0] * L
    qd = [0] * L
    pd_bus = [0] * N
    qd_bus = [0] * N
    for i, key in enumerate(sorted_load_keys):
        pd[i] = network['data']['load'][key]['pd']
        qd[i] = network['data']['load'][key]['qd']
        bus_id = network['data']['load'][key]['source_id'][1]
        bus_index = bus_id_to_index[bus_id]
        pd_bus[bus_index] = pd[i]
        qd_bus[bus_index] = qd[i]
    # # Very robust :)
    # min_perc = 0.9999
    # max_perc = 1.0001
    # random_perc = 0.00001
    min_perc = 0.999
    max_perc = 1.001
    random_perc = 0.0001

    with open(f"vnnlib/{network_name}_prop1.vnnlib", 'w') as f:
        # check power balance constraints violation
        f.write("; Check power balance violation:\n")
        # declare constants
        for x in range(input_shape[1]):
            f.write(f"(declare-const X_{x} Real)\n")
        f.write("\n")
        for x in range(output_shape[1]):
            f.write(f"(declare-const Y_{x} Real)\n")
        f.write("\n")
        f.write("; Input constraints:\n")
        # input perturbation
        perturbation = [random.uniform(-random_perc, random_perc) for i in range(L)]  # generate a list of random numbers between -0.01 and 0.01
        for i in range(L):
            lb = pd[i] * min_perc
            ub = pd[i] * max_perc
            if pd[i] < 0:
                lb = pd[i] * max_perc
                ub = pd[i] * min_perc
            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])   # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        for i in range(L):
            lb = qd[i] * min_perc
            ub = qd[i] * max_perc
            # update lb and ub for negative values
            if qd[i] < 0:
                lb = qd[i] * max_perc
                ub = qd[i] * min_perc
            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i+L} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i+L} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        # output properties
        f.write("; Output property:\n")
        for i in range(N):
            ub = max(10**(-3), 10**(-2)*pd_bus[i])
            lb = -ub
            f.write(f"(assert (<= Y_{i+output_shape[1]-2*N} {round(ub, 9)}))\n")
            f.write(f"(assert (>= Y_{i+output_shape[1]-2*N} {round(lb, 9)}))\n")
        for i in range(N):
            ub = max(10**(-3), 10**(-2)*qd_bus[i])
            lb = -ub
            f.write(f"(assert (<= Y_{i+output_shape[1]-N} {round(ub, 9)}))\n")
            f.write(f"(assert (>= Y_{i+output_shape[1]-N} {round(lb, 9)}))\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    # call main function with the network name argument
    network_names = ["14_ieee", "300_ieee"]
    for network_name in network_names:
        # generate vnnlib files
        main(network_name, args.seed)

    # generate instances.csv file
    timeout = 300
    csvFile = open("instances.csv", "w")
    for network in os.listdir('onnx'):
        for prop in os.listdir('vnnlib'):
            if "_".join(network.split("_")[:2]) == "_".join(prop.split("_")[:2]):
                print(f"onnx/{network},vnnlib/{prop},{timeout}", file=csvFile)
    csvFile.close()