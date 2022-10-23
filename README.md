# Data Mining
## Lab1
### Download the synthetic data generator (IBMGenerator) from IBM Almaden Quest Project
    git clone https://github.com/zakimjz/IBMGenerator.git
### Generate the executable file 'gen'
    make
### Generate the datasets 'A.data', 'B.data', 'C.data'
    ./gen lit -ntrans 1 -tlen 10 -nitems 0.5 -fname A -ascii
    ./gen lit -ntrans 100 -tlen 10 -nitems 1 -fname B -ascii
    ./gen lit -ntrans 1000 -tlen 10 -nitems 1 -fname C -ascii
### Reference
    [https://github.com/chonyy/fpgrowth_py](https://github.com/chonyy/fpgrowth_py)
