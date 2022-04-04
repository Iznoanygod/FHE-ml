#include "fheml.h"
#include <iostream>
#include <fstream>

int main() {
    std::string line;
    ml::Network *net = new ml::Network(784,200,10,0.005);
    net->randomize_weights();
    std::ifstream test_file("../mnist_train.csv");
    net->load("net.nf");
    std::cout << "created network" << std::endl;
    ml::FHENetwork *fhenet = new ml::FHENetwork(net);
    std::cout << "created encrypted network" << std::endl;
    auto cc = fhenet->get_cc();
    auto key = fhenet->get_key();
    
    while(getline(test_file, line)) {
        std::stringstream s_stream(line);
        std::string tar;
        getline(s_stream, tar, ',');
        int target = std::stoi(tar);

        vector<double> target_vector;
        target_vector.resize(10);
        target_vector[target] = 1.0;

        vector<double> input;
        input.resize(784);
        for(int i = 0; i < 784; i++) {
            std::string val;
            getline(s_stream, tar, ',');
            int ta = std::stoi(tar);
            input[i] = ta/255.;
        }
        
        Plaintext pinput = cc->MakeCKKSPackedPlaintext(input);
        Ciphertext<DCRTPoly> cinput = cc->Encrypt(key.publicKey, pinput);
        
        vector<double> standard_predict = net->predict(input);
        Ciphertext<DCRTPoly> encrypt_predict = fhenet->predict(cinput);
        std::cout << "Prediction done" << std::endl;
        int max = 0;
        for(int i = 1; i < 10; i++)
            if(standard_predict[i] > standard_predict[max])
                max = i;
        Plaintext decrypt;
        cc->Decrypt(key.secretKey, encrypt_predict, &decrypt);
        decrypt->SetLength(10);
        vector<double> uvec = decrypt->GetRealPackedValue();
        int cmax = 0;
        for(int i = 1; i < 10; i++)
            if(uvec[i] > uvec[cmax])
                cmax = i;
        std::cout << "Target:" +std::to_string(target) + " | Prediction:" + std::to_string(max)  + " | Encrypted:" + std::to_string(cmax) << std::endl;

    }
}
