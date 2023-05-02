# ml4acopf_benchmark
Benchmark of ml4acopf for VNN-COMP 2023

## Links
- VNN-COMP 2023 competition website: https://sites.google.com/view/vnn2023
- VNN-COMP 2023 competition GitHub repo: https://github.com/stanleybak/vnncomp2023
- VNN-COMP 2022 benchmarks: https://github.com/ChristopherBrix/vnncomp2022_benchmarks

## Proposing a new benchmark
Source: https://github.com/stanleybak/vnncomp2023/issues/2
>To propose a new benchmark, please create a public git repository with all the necessary code.
>The repository must be structured as follows:
> - It must contain a generate_properties.py file which accepts the seed as the only command line parameter.
> - There must be a folder with all .vnnlib files, which may be identical to the folder containing the generate_properties.py file
> - There must be a folder with all .onnx files, which may be identical to the folder containing the generate_properties.py file
> - The generate_properties.py file will be run using Python 3.8 on a t2.large AWS instance. (see https://vnncomp.christopher-brix.de/)
