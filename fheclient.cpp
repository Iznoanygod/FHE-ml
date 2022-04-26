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
        //vector<double> predict = net->predict(input);
        //int max = 0;
        //for(int i = 1; i < 10; i++)
        //    if(predict[i] > predict[max])
        //        max = i;
        //if(max == target){
        //    correct++;
        //}
        //net->train(input, target_vector);
        //i++;
        //if(i % 1000 == 0)
        //    std::cout << i << std::endl;
        
        Plaintext pinput = cc->MakeCKKSPackedPlaintext(input);
        Ciphertext<DCRTPoly> cinput = cc->Encrypt(key.publicKey, pinput);
        
        vector<double> standard_predict = net->predict(input);
        Ciphertext<DCRTPoly> encrypt_predict = fhenet->first_predict(cinput);
        Plaintext in_between;
        cc->Decrypt(key.secretKey, encrypt_predict, &in_between);
        in_between->SetLength(200);
        vector<double> in_b = in_between->GetRealPackedValue();
        for(int k = 0; k < 200; k++) {
            in_b[k] = 1 / (1 + exp(-in_b[k]));
        }
        in_between = cc->MakeCKKSPackedPlaintext(in_b);
        encrypt_predict = cc->Encrypt(key.publicKey, in_between);
        //encrypt_predict = cc->EvalPoly(encrypt_predict, {0.5, 0.164128, 0, -0.00260371, 0, 0.000014906});
        encrypt_predict = fhenet->second_predict(encrypt_predict);
        std::cout << "Prediction done" << std::endl;
        int max = 0;
        for(int i = 1; i < 10; i++)
            if(standard_predict[i] > standard_predict[max])
                max = i;
        Plaintext decrypt;
        cc->Decrypt(key.secretKey, encrypt_predict, &decrypt);
        decrypt->SetLength(10);
        std::cout << decrypt << std::endl;
        vector<double> uvec = decrypt->GetRealPackedValue();
        int cmax = 0;
        for(int i = 1; i < 10; i++)
            if(uvec[i] > uvec[cmax])
                cmax = i;
        std::cout << "Target:" +std::to_string(target) + " | Prediction:" + std::to_string(max)  + " | Encrypted:" + std::to_string(cmax) << std::endl;

    }
    //printf("%d\n", correct);
    //net->save("net.nf");
}
