#include "fheml.h"
#include <iostream>
#include <fstream>
#include "bmpparse.h"

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "scheme/ckks/ckks-ser.h"
#include "pubkeylp-ser.h"

int main(int argc, char **argv) {
    if(!strcmp(argv[1], "train")) {
        ml::Network *net = new ml::Network(784, 200, 10, 0.005);
        net->load(argv[2]);
        int epoch = atoi(argv[3]);
        std::string line;
        int total = 0;
        for(int i = 0; i < epoch; i++) {
            std::ifstream test_file("../mnist_train.csv");
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
                    input[i] = ta / 255.;
                }
                net->train(input, target_vector);
                total++;
                if(total % 1000 == 0)
                    printf("%d\n", total);
            }
        }
        net->save(argv[2]);
    }
    else if(!strcmp(argv[1], "create")) {
        ml::Network *net = new ml::Network(784, 200, 10, 0.005);
        net->randomize_weights();
        net->save(argv[2]);
    }
    else if(!strcmp(argv[1], "test")) {
        ml::Network *net = new ml::Network(784, 200, 10, 0.005);
        net->load(argv[2]);
        std::string line;
        std::ifstream test_file("../mnist_test.csv");
        int correct = 0;
        int total = 0;
        while(getline(test_file, line)) {
            std::stringstream s_stream(line);
            std::string tar;
            getline(s_stream, tar, ',');
            int target = std::stoi(tar);

            vector<double> input;
            input.resize(784);
            for(int i = 0; i < 784; i++) {
                std::string val;
                getline(s_stream, tar, ',');
                int ta = std::stoi(tar);
            //input->set(i,0,ta / 255.);
                input[i] = ta / 255.;
            }
            vector<double> result = net->predict(input);
            int max = 0;
            for(int i = 1; i < 10; i++)
                if(result[i] > result[max])
                    max = i;
            if(max == target)
                correct++;
            total++;
        }
        printf("%d/%d\n", correct, total);
    }
    else if(!strcmp(argv[1], "encrypt")) {
        CryptoContext<DCRTPoly> cc;
        Serial::DeserializeFromFile("cryptocontext.cf", cc, SerType::BINARY);
        LPKeyPair<DCRTPoly> keys;
        Serial::DeserializeFromFile("public.kf", keys.publicKey, SerType::BINARY);
        Serial::DeserializeFromFile("private.kf", keys.secretKey, SerType::BINARY);
        ml::Network *net = new ml::Network(784, 200, 10, 0.005);
        net->load(argv[2]);
        std::string weight_dir = argv[3];
        ml::FHENetwork *fhenet = new ml::FHENetwork(net, cc, keys);
        fhenet->save(weight_dir);
    }
}
