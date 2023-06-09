import random
import onnxruntime  as ort
import json
import os
import csv
import sys

def extract_branch_limit(br_data):
    rate_A = br_data.get("rate_a", 0.0)

    # unbounded if rate_A is 0 by convention
    rate_A += 1e12 * (rate_A == 0)

    Smax = rate_A
    # Convert thermal limits into current limits
    return Smax

def read_csv_data(property_number, network_name, file_path='data/vnnlib_params.csv'):
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) >= 5 and row[0] == property_number and row[1] == network_name:
                min_perc = float(row[2])
                max_perc = float(row[3])
                random_perc = float(row[4])
                return min_perc, max_perc, random_perc

    return 0.95, 1.05, 0.001  # Return default values if the data is not found

# test power balance violation
def generate_vnnlib_file_prop1(network, network_name, input_shape, output_shape):
    # Sort the keys of the load data dictionary
    sorted_load_keys = sorted(network['data']['load'].keys(), key=lambda x: int(x))
    sorted_bus_keys = sorted(network['data']['bus'].keys(), key=lambda x: int(x))

    bus_id_to_index = {}
    for idx, key in enumerate(sorted_bus_keys):
        bus_id = int(network['data']['bus'][key]['source_id'][1])
        bus_id_to_index[bus_id] = idx

    # Reference load
    N = len(network['data']['bus'])
    L = len(network['data']['load'])
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

    min_perc, max_perc, random_perc = read_csv_data('1', network_name)

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
        perturbation = [random.uniform(-random_perc, random_perc) for i in range(L)]  # generate a list of random numbers
        for i in range(L):
            lb = pd[i] * min_perc if pd[i] >= 0 else pd[i] * max_perc
            ub = pd[i] * max_perc if pd[i] >= 0 else pd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        for i in range(L):
            lb = qd[i] * min_perc if qd[i] >= 0 else qd[i] * max_perc
            ub = qd[i] * max_perc if qd[i] >= 0 else qd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i+L} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i+L} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        # output properties
        f.write("; Output property:\n")
        f.write("(assert (or\n")
        for i in range(N):
            ub = max(10**(-3), 10**(-2)*pd_bus[i])
            lb = -ub
            f.write(f"(and (>= Y_{i+output_shape[1]-2*N} {round(ub, 9)}))\n")
            f.write(f"(and (<= Y_{i+output_shape[1]-2*N} {round(lb, 9)}))\n")
        for i in range(N):
            ub = max(10**(-3), 10**(-2)*qd_bus[i])
            lb = -ub
            f.write(f"(and (>= Y_{i+output_shape[1]-N} {round(ub, 9)}))\n")
            f.write(f"(and (<= Y_{i+output_shape[1]-N} {round(lb, 9)}))\n")
        f.write("))\n")

# test thermal limit violation
def generate_vnnlib_file_prop2(network, network_name, input_shape, output_shape):
    # Sort the keys of the load data dictionary
    sorted_load_keys = sorted(network['data']['load'].keys(), key=lambda x: int(x))
    sorted_branch_keys = sorted(network['data']['branch'].keys(), key=lambda x: int(x))

    # Reference load
    N = len(network['data']['bus'])
    L = len(network['data']['load'])
    E = len(network['data']['branch'])
    pd = [0] * L
    qd = [0] * L
    Smax = [0] * E
    for i, key in enumerate(sorted_load_keys):
        pd[i] = network['data']['load'][key]['pd']
        qd[i] = network['data']['load'][key]['qd']

    for e, key in enumerate(sorted_branch_keys):
        Smax[e] = extract_branch_limit(network['data']['branch'][key])

    # choose k-th largest element in the list
    k = 5
    sorted_Smax = sorted(Smax, reverse=True)
    line_idx = 0
    if len(sorted_Smax) >= k:
        kth_largest = sorted_Smax[k-1]
        line_idx = Smax.index(kth_largest)

    min_perc, max_perc, random_perc = read_csv_data('2', network_name)

    with open(f"vnnlib/{network_name}_prop2.vnnlib", 'w') as f:
        # check thermal limits violation
        f.write("; Check thermal limits violation:\n")
        # declare constants
        for x in range(input_shape[1]):
            f.write(f"(declare-const X_{x} Real)\n")
        f.write("\n")
        for x in range(output_shape[1]):
            f.write(f"(declare-const Y_{x} Real)\n")
        f.write("\n")
        f.write("; Input constraints:\n")
        # input perturbation
        perturbation = [random.uniform(-random_perc, random_perc) for i in range(L)]  # generate a list of random numbers
        for i in range(L):
            lb = pd[i] * min_perc if pd[i] >= 0 else pd[i] * max_perc
            ub = pd[i] * max_perc if pd[i] >= 0 else pd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        for i in range(L):
            lb = qd[i] * min_perc if qd[i] >= 0 else qd[i] * max_perc
            ub = qd[i] * max_perc if qd[i] >= 0 else qd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i+L} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i+L} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        # output properties
        f.write("; Output property:\n")
        ub = max(10**(-2), 10**(-2)*Smax[line_idx])
        f.write(f"(assert (>= Y_{line_idx+output_shape[1]-2*N-2*E} {round(ub, 9)}))\n")
    return

# test pg/qg bound violation
def generate_vnnlib_file_prop3(network, network_name, input_shape, output_shape):
    # Sort the keys of the load data dictionary
    sorted_load_keys = sorted(network['data']['load'].keys(), key=lambda x: int(x))
    sorted_gen_keys = sorted(network['data']['gen'].keys(), key=lambda x: int(x))

    # Reference load
    N = len(network['data']['bus'])
    L = len(network['data']['load'])
    G = len(network['data']['gen'])
    pd = [0] * L
    qd = [0] * L
    pmax = [0] * G
    pmin = [0] * G
    qmax = [0] * G
    qmin = [0] * G

    for i, key in enumerate(sorted_load_keys):
        pd[i] = network['data']['load'][key]['pd']
        qd[i] = network['data']['load'][key]['qd']

    for g, key in enumerate(sorted_gen_keys):
        pmax[g] = network['data']['gen'][key]['pmax']
        pmin[g] = network['data']['gen'][key]['pmin']
        qmax[g] = network['data']['gen'][key]['qmax']
        qmin[g] = network['data']['gen'][key]['qmin']

    min_perc, max_perc, random_perc = read_csv_data('3', network_name)

    output_epsilon = 1e-06

    with open(f"vnnlib/{network_name}_prop3.vnnlib", 'w') as f:
        # check generation bounds violation
        f.write("; Check generation bounds violation:\n")
        # declare constants
        for x in range(input_shape[1]):
            f.write(f"(declare-const X_{x} Real)\n")
        f.write("\n")
        for x in range(output_shape[1]):
            f.write(f"(declare-const Y_{x} Real)\n")
        f.write("\n")
        f.write("; Input constraints:\n")
        # input perturbation
        perturbation = [random.uniform(-random_perc, random_perc) for i in range(L)]  # generate a list of random numbers
        for i in range(L):
            lb = pd[i] * min_perc if pd[i] >= 0 else pd[i] * max_perc
            ub = pd[i] * max_perc if pd[i] >= 0 else pd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        for i in range(L):
            lb = qd[i] * min_perc if qd[i] >= 0 else qd[i] * max_perc
            ub = qd[i] * max_perc if qd[i] >= 0 else qd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i+L} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i+L} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        # output properties
        f.write("; Output property:\n")
        f.write("(assert (or\n")
        for g in range(G):
            ub = pmax[g] + output_epsilon
            lb = pmin[g] - output_epsilon
            f.write(f"(and (>= Y_{g} {ub:.9f}))\n")
            f.write(f"(and (<= Y_{g} {lb:.9f}))\n")
        for g in range(G):
            ub = qmax[g] + output_epsilon
            lb = qmin[g] - output_epsilon
            f.write(f"(and (>= Y_{g+G} {ub:.9f}))\n")
            f.write(f"(and (<= Y_{g+G} {lb:.9f}))\n")
        f.write("))\n")
    return

# test vm bound violation
def generate_vnnlib_file_prop4(network, network_name, input_shape, output_shape):
    # Sort the keys of the load data dictionary
    sorted_load_keys = sorted(network['data']['load'].keys(), key=lambda x: int(x))
    sorted_bus_keys = sorted(network['data']['bus'].keys(), key=lambda x: int(x))

    # Reference load
    N = len(network['data']['bus'])
    L = len(network['data']['load'])
    G = len(network['data']['gen'])
    pd = [0] * L
    qd = [0] * L
    vmax = [0] * N
    vmin = [0] * N

    for i, key in enumerate(sorted_load_keys):
        pd[i] = network['data']['load'][key]['pd']
        qd[i] = network['data']['load'][key]['qd']

    for i, key in enumerate(sorted_bus_keys):
        vmax[i] = network['data']['bus'][key]['vmax']
        vmin[i] = network['data']['bus'][key]['vmin']

    min_perc, max_perc, random_perc = read_csv_data('4', network_name)

    output_epsilon = 1e-06

    with open(f"vnnlib/{network_name}_prop4.vnnlib", 'w') as f:
        # check generation bounds violation
        f.write("; Check generation bounds violation:\n")
        # declare constants
        for x in range(input_shape[1]):
            f.write(f"(declare-const X_{x} Real)\n")
        f.write("\n")
        for x in range(output_shape[1]):
            f.write(f"(declare-const Y_{x} Real)\n")
        f.write("\n")
        f.write("; Input constraints:\n")
        # input perturbation
        perturbation = [random.uniform(-random_perc, random_perc) for i in range(L)]  # generate a list of random numbers
        for i in range(L):
            lb = pd[i] * min_perc if pd[i] >= 0 else pd[i] * max_perc
            ub = pd[i] * max_perc if pd[i] >= 0 else pd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        for i in range(L):
            lb = qd[i] * min_perc if qd[i] >= 0 else qd[i] * max_perc
            ub = qd[i] * max_perc if qd[i] >= 0 else qd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i+L} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i+L} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        # output properties
        f.write("; Output property:\n")
        f.write("(assert (or\n")
        for i in range(N):
            ub = vmax[i] + output_epsilon
            lb = vmin[i] - output_epsilon
            f.write(f"(and (>= Y_{i+2*G} {ub:.9f}))\n")
            f.write(f"(and (<= Y_{i+2*G} {lb:.9f}))\n")
        f.write("))\n")
    return

# test single bus power balance violation
def generate_vnnlib_file_prop_pb_single(network, network_name, input_shape, output_shape, k):
    # Sort the keys of the load data dictionary
    sorted_load_keys = sorted(network['data']['load'].keys(), key=lambda x: int(x))
    sorted_bus_keys = sorted(network['data']['bus'].keys(), key=lambda x: int(x))

    bus_id_to_index = {}
    for idx, key in enumerate(sorted_bus_keys):
        bus_id = int(network['data']['bus'][key]['source_id'][1])
        bus_id_to_index[bus_id] = idx

    # Reference load
    sample_load_path = os.path.join("data", f"{network_name}_sample.json")
    with open(sample_load_path, 'r') as file:
        # Load the data from the file
        sample_load = json.load(file)
    N = len(network['data']['bus'])
    L = len(network['data']['load'])
    pd = [0] * L
    qd = [0] * L
    pd_bus = [0] * N
    qd_bus = [0] * N
    for i, key in enumerate(sorted_load_keys):
        pd[i] = sample_load['pd'][i]
        qd[i] = sample_load['qd'][i]
        bus_id = network['data']['load'][key]['source_id'][1]
        bus_index = bus_id_to_index[bus_id]
        pd_bus[bus_index] = pd[i]
        qd_bus[bus_index] = qd[i]

    # only check one index for p_balance violation
    sorted_pd_bus = sorted(pd_bus, reverse=True)

    pindex = pd_bus.index(max(pd_bus))

    if len(sorted_pd_bus) >= k:
        kth_largest = sorted_pd_bus[k-1]
        pindex = pd_bus.index(kth_largest)

    min_perc, max_perc, random_perc = read_csv_data('5', network_name)

    with open(f"vnnlib/{network_name}_prop{5+k}.vnnlib", 'w') as f:
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
        perturbation = [random.uniform(-random_perc, random_perc) for i in range(L)]  # generate a list of random numbers
        for i in range(L):
            lb = pd[i] * min_perc if pd[i] >= 0 else pd[i] * max_perc
            ub = pd[i] * max_perc if pd[i] >= 0 else pd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        for i in range(L):
            lb = qd[i] * min_perc if qd[i] >= 0 else qd[i] * max_perc
            ub = qd[i] * max_perc if qd[i] >= 0 else qd[i] * min_perc

            perturbed_lb = lb * (1 + perturbation[i])  # add the perturbation to the original lb
            perturbed_ub = ub * (1 + perturbation[i])  # add the perturbation to the original ub
            f.write(f"(assert (<= X_{i+L} {round(perturbed_ub, 9)}))\n")
            f.write(f"(assert (>= X_{i+L} {round(perturbed_lb, 9)}))\n")
            f.write("\n")
        
        # output properties
        f.write("; Output property:\n")
        f.write("(assert (or\n")
        ub = max(10**(-2), 10**(-2)*pd_bus[pindex])
        lb = -ub
        f.write(f"(and (>= Y_{pindex+output_shape[1]-2*N} {round(ub, 9)}))\n")
        f.write(f"(and (<= Y_{pindex+output_shape[1]-2*N} {round(lb, 9)}))\n")
        f.write("))\n")

def main(network_name, seed):
    random.seed(seed)  # set a specific seed value for reproducibility

    data_folder = "data"
    network_path = os.path.join(data_folder, f"{network_name}.ref.json")

    # Open the JSON file
    with open(network_path, 'r') as file:
        # Load the data from the file
        network = json.load(file)

    model_path = os.path.join("onnx", f"{network_name}_ml4acopf.onnx")

    # Load the ONNX model into memory
    sess = ort.InferenceSession(model_path)

    # Get information about the input and output nodes of the ONNX model
    input_info = sess.get_inputs()
    output_info = sess.get_outputs()

    # Assume the first input and output nodes are the ones you want to use
    input_shape = input_info[0].shape
    output_shape = output_info[0].shape

    if network_name=="14_ieee":
        generate_vnnlib_file_prop1(network, network_name, input_shape, output_shape)
    generate_vnnlib_file_prop2(network, network_name, input_shape, output_shape)
    generate_vnnlib_file_prop3(network, network_name, input_shape, output_shape)
    generate_vnnlib_file_prop4(network, network_name, input_shape, output_shape)
    if network_name=="14_ieee":
        for k in range(10):
            generate_vnnlib_file_prop_pb_single(network, network_name, input_shape, output_shape, k)
    elif network_name=="118_ieee":
        for k in range(2):
            generate_vnnlib_file_prop_pb_single(network, network_name, input_shape, output_shape, k)
    elif network_name=="300_ieee":
        for k in [100]:
            # for fun
            generate_vnnlib_file_prop_pb_single(network, network_name, input_shape, output_shape, k)

if __name__ == '__main__':
    seed=42
    # if the seed value is provided
    if len(sys.argv) == 2:
        seed = int(sys.argv[1])

    # call main function with the network name argument
    network_names = ["14_ieee", "118_ieee","300_ieee"]
    for network_name in network_names:
        # generate vnnlib files
        main(network_name, seed)

    # generate instances.csv file
    timeout = 600
    csvFile = open("instances.csv", "w")
    for network in os.listdir('onnx'):
        for prop in os.listdir('vnnlib'):
            if "_".join(network.split("_")[:2]) == "_".join(prop.split("_")[:2]):
                print(f"./onnx/{network},./vnnlib/{prop},{timeout}", file=csvFile)
    csvFile.close()