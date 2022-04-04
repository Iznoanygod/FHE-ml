#include "fheml.h"
#include <math.h>
#include <fstream>

int main(int argc, char** argv) {
    // TODO add cryptocontexts of different batch sizes

    //SecurityLevel securityLevel = HEStd_128_classic;

    usint m = 8192;

    usint init_size = 3;
    usint dcrtBits = 40;

    CryptoContext<DCRTPoly> cc =
        CryptoContextFactory<DCRTPoly>::genCryptoContextCKKSWithParamsGen(
            m, init_size, /*numPrimes*/
            dcrtBits, 10, /*relinWindow*/
            784,           /*batch size*/
            OPTIMIZED, 4 /*depth*/);

	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    
    cc->EvalMultKeyGen(keys.secretKey);
    
    cc->EvalSumKeyGen(keys.secretKey);

    /*vector<double> vector1 = {1,2,3,4};
    vector<double> vector2 = {2,3,4,5};
    vector<double> vector3 = {3,4,5,6};
    vector<double> vector4 = {4,5,6,7};

    vector<double> vectormultiple = {2,4,6,8};

    Plaintext pv1 = cc->MakeCKKSPackedPlaintext(vector1);
    Plaintext pv2 = cc->MakeCKKSPackedPlaintext(vector2);
    Plaintext pv3 = cc->MakeCKKSPackedPlaintext(vector3);
    Plaintext pv4 = cc->MakeCKKSPackedPlaintext(vector4);
    Plaintext pvm = cc->MakeCKKSPackedPlaintext(vectormultiple);


    Ciphertext<DCRTPoly> ev1 = cc->Encrypt(keys.publicKey, pv1);
    Ciphertext<DCRTPoly> ev2 = cc->Encrypt(keys.publicKey, pv2);
    Ciphertext<DCRTPoly> ev3 = cc->Encrypt(keys.publicKey, pv3);
    Ciphertext<DCRTPoly> ev4 = cc->Encrypt(keys.publicKey, pv4);
    Ciphertext<DCRTPoly> evm = cc->Encrypt(keys.publicKey, pvm);

    //vector<Ciphertext<DCRTPoly>> ciphertexts;
    //cc->EvalAtIndexKeyGen(keys.secretKey, {-1,-2,-3,-4});

    //ciphertexts.push_back(cc->EvalMult(ev1, evm));
    //ciphertexts.push_back(cc->EvalMult(ev2, evm));
    //ciphertexts.push_back(cc->EvalMult(ev3, evm));
    //ciphertexts.push_back(cc->EvalMult(ev4, evm));
    auto a = cc->EvalMult(ev1, ev2);
    auto c = cc->EvalMult(ev3, 1);
    auto b = cc->EvalAdd(a, c);
    //auto merged = cc->EvalMerge(ciphertexts);
    //std::cout << "Merged" << std::endl;
    //auto merged_add = cc->EvalAdd(merged, evm);
    Plaintext merged_plaintext;
    cc->Decrypt(keys.secretKey, b, &merged_plaintext);

    merged_plaintext->SetLength(4);
    std::cout << *merged_plaintext << std::endl;
    */
    vector<int> rotations(200);
    for(int i = 1; i < 201; i++)
        rotations[i] = -i;
    cc->EvalAtIndexKeyGen(keys.secretKey, rotations);
    std::cout << "setup" << std::endl;
    
    //vector<double> vectxt = {0.1, 0.2, 0.3};

    //auto ptxt = cc->MakeCKKSPackedPlaintext(vectxt);

    //auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    //std::cout << sizeof(ctxt) << std::endl;
    /*fhe::Matrix *M = new fhe::Matrix(2,2);
    M->set(0,0,1);
    M->set(0,1,2);
    M->set(1,0,3);
    M->set(1,1,4);

    std::cout << M->toString() << std::endl;
    
    fhe::FHEMatrix *M_e = new fhe::FHEMatrix(M,cc, keys);
    fhe::FHEMatrix *MM_e = M_e->multiply(M_e);
    fhe::Matrix *M_d = MM_e->decrypt(keys);

    std::cout << M_d->toString() << std::endl;

    delete M_e;
    delete MM_e;
    delete M;
    delete M_d;
    */
    ml::Network *net = new ml::Network(784,200,10,0.005);
    net->randomize_weights();
    net->save("net.nf");
    //net->load("net1.nf");
    std::string line;
    ml::FHENetwork *fhenet = new ml::FHENetwork(net, cc, keys);
    std::cout << "Created Network" << std::endl;
    
    /*for(int epochs = 0; epochs < 1; epochs++){
    std::ifstream training_file("../mnist_train.csv");
    int t_cout = 0;
    while(getline(training_file, line)) {
        //std::cout << line << std::endl; 
        std::stringstream s_stream(line);
        std::string tar;
        getline(s_stream, tar, ',');
        int target = std::stoi(tar);
        
        fhe::Matrix *target_vector = new fhe::Matrix(10,1);
        for(int i = 0; i < 10; i++)
            target_vector->set(i,0,0);
        target_vector->set(target,0,1.0);

        fhe::Matrix *input = new fhe::Matrix(784,1);
        
        for(int i = 0; i < 784; i++) {
            std::string val;
            getline(s_stream, tar, ',');
            int ta = std::stoi(tar);
            input->set(i,0,ta / 255.);
        }
        net->train(input, target_vector); 
        delete input;
        delete target_vector;
        if(t_cout % 1000 == 0) {
            std::cout << t_cout << std::endl;
        }
        t_cout++;
    }
    }*/
    std::ifstream test_file("../mnist_test.csv");
    int correct = 0;
    int total = 0;
    while(getline(test_file, line)) {
        std::stringstream s_stream(line);
        std::string tar;
        getline(s_stream, tar, ',');
        int target = std::stoi(tar);
        
        vector<double> target_vector;
        target_vector.resize(10);
        target_vector[target] = 1.0;
        //fhe::Matrix *target_vector = new fhe::Matrix(10,1);
        //for(int i = 0; i < 10; i++)
        //    target_vector->set(i,0,0);
        //target_vector->set(target,0,1.0);

        //fhe::Matrix *input = new fhe::Matrix(784,1);
        vector<double> input;
        input.resize(784);
        for(int i = 0; i < 784; i++) {
            std::string val;
            getline(s_stream, tar, ',');
            int ta = std::stoi(tar);
            //input->set(i,0,ta / 255.);
            input[i] = ta / 255.;
        }

        Plaintext plaininput = cc->MakeCKKSPackedPlaintext(input);
        Ciphertext<DCRTPoly> input_e = cc->Encrypt(keys.publicKey, plaininput);

        std::cout << "Performing inference" << std::endl;
        Ciphertext<DCRTPoly> prediction = fhenet->predict_first_layer(input_e);
        
        Plaintext hidden_layer;
        cc->Decrypt(keys.secretKey, prediction, &hidden_layer);
        hidden_layer->SetLength(200);
        vector<double> hidden_vector = hidden_layer->GetRealPackedValue();
        for(int i = 0; i < 200; i++){
            hidden_vector[i] = 1. / (1 + exp(-hidden_vector[i]));
        }

        hidden_layer = cc->MakeCKKSPackedPlaintext(hidden_vector);
        prediction = cc->Encrypt(keys.publicKey, hidden_layer);
        Ciphertext<DCRTPoly> prediction_final = fhenet->predict_second_layer(prediction);


        Plaintext result;
        cc->Decrypt(keys.secretKey, prediction_final, &result);
        result->SetLength(10);
        vector<double> resultvector = result->GetRealPackedValue();
        for(int i = 0; i < 10; i++) {
            resultvector[i] = 1. / (1 + exp(-resultvector[i]));
        }
        std::cout << result << std::endl;
        //std::cout << prediction->toString() << std::endl;
        int max = 0;
        for(int i = 0; i < 10; i++)
            if(resultvector[i] > resultvector[max])
                max = i;
        std::cout << "predicted:"+std::to_string(max)+" | target:"+std::to_string(target) <<std::endl;
        total++;
        if(max == target)
            correct++;
    }
    double percent = correct * 100. / total;
    std::cout << percent <<std::endl;
    
    //net->save("net1.nf");
    
    return 0;
}
