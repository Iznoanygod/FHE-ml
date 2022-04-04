#include "fheml.h"
#include <iostream>
#include <fstream>

int main() {
    std::string line;
    ml::Network *net = new ml::Network(784,200,10,0.005);
    net->randomize_weights();
    std::ifstream test_file("../mnist_train.csv");
    net->load("net.nf");
    ml::FHENetwork *fhenet = new ml::FHENetwork(net);
    auto cc = fhenet->get_cc();
}
