# Benchmark
The benchmark files are located in the `14_ieee` folder:
- onnx file: `ldf14bus.onnx`
- vnnlib file: `14_bus_prop1.vnnlib`
---
## Parameters
- N = 14 : number of buses
- G =  5 : number of generators
- L = 11 : number of loads
- E = 20 : number of lines
- B = 32 : batch size
---
## Onnx model description
### Input: `pd/qd`
dim: (B, 2L) = (32, 22)
- `pd`: Real power demand. (B, L)
- `qd`: Reactive power demand. (B, L)
### Output: `pg/qg/vm/va/pf/pt/qf/qt/thrm_1/thrm_2/p_balance/q_balance`
dim: (B, 2G + 4N + 6E) = (32, 186)
- `pg`: Real power generation. (B, G)
- `qg`: Reactive power generation. (B, G)
- `vm`: Voltage magnitude. (B, N)
- `va`: Voltage angle. (B, N)
- `pf`: Real power flow from. (B, E)
- `pt`: Real power flow to. (B, E)
- `qf`: Reactive power flow from. (B, E)
- `qt`: Reactive power flow to. (B, E)
- `thrm_1`: Thermal limit from violation. (B, E)
- `thrm_2`: Thermal limit to violation. (B, E)
- `p_balance`: Real power balance violation. (B, N)
- `q_balance`: Reactive power balance violation. (B, N)
---
## Vnnlib description
### Input:
+- 5% perturbation of the reference active and reactive load 
### Output:
Check power balance violation:

for each bus `i`
- `|p_balance[i]| <= max(10^(-3), 10^(-2)*pd_i)`
- `|q_balance[i]| <= max(10^(-3), 10^(-2)*qd_i)`

---
## Inference
The code to run inference is presented in the `main.py` file.
### Environment Setup
To set up the environment, follow these steps:
1. Create the environment using the command: `conda env create -f env.yaml`
2. Activate the environment using the command: `conda activate onnx-vnnlib-env`