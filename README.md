# Data Mining
## Lab1
1. Download the synthetic data generator (IBMGenerator) from IBM Almaden Quest Project.

    git clone https://github.com/zakimjz/IBMGenerator.git

2. Generate the executable file 'gen'

    make
    
3. Generate the datasets 'A.data', 'B.data', 'C.data'

    ./gen lit -ntrans 1 -tlen 10 -nitems 0.5 -fname A -ascii
    ./gen lit -ntrans 100 -tlen 10 -nitems 1 -fname B -ascii
    ./gen lit -ntrans 1000 -tlen 10 -nitems 1 -fname C -ascii
