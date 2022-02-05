#include "fheml.h"
<<<<<<< HEAD

int main(int argc, char** argv) {
    uint32_t multDepth = 1;
    uint32_t scaleFactorBits = 50;
    uint32_t batchSize = 2;
=======
#include <fstream>

int main(int argc, char** argv) {
    /*uint32_t multDepth = 8;
    uint32_t scaleFactorBits = 50;
    uint32_t batchSize = 1;
>>>>>>> 77aed8b (02.05.22)

    SecurityLevel securityLevel = HEStd_128_classic;

	CryptoContext<DCRTPoly> cc =
		CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
			multDepth, scaleFactorBits, batchSize, securityLevel);

	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);

    auto keys = cc->KeyGen();
	
    cc->EvalMultKeyGen(keys.secretKey);

<<<<<<< HEAD
    fhe::Matrix M(2,2,1);
    M[0][0] = {1};
    M[0][1] = {2};
    M[1][0] = {3};
    M[1][1] = {4};

    std::cout << std::string(M) << std::endl;

    fhe::Matrix N(2,2,2);
    N[0][0] = {1,2};
    N[0][1] = {2,4};
    N[1][0] = {3,6};
    N[1][1] = {4,8};

    std::cout << std::string(N) << std::endl;

    fhe::Matrix MpN = M+N;
    fhe::Matrix MxN = M*N;

    std::cout << std::string(MpN) << std::endl;
    std::cout << std::string(MxN) << std::endl;

    std::cout << "FHE" << std::endl;
    fhe::FHEMatrix M_e(M,2,cc,keys);
    fhe::FHEMatrix N_e(N,2,cc,keys);

    fhe::FHEMatrix MpN_e = M_e+N_e;
    fhe::FHEMatrix MxN_e = M_e*N_e;

    fhe::Matrix M_d = M_e.decrypt(keys);
    fhe::Matrix N_d = N_e.decrypt(keys);

    fhe::Matrix MpN_d = MpN_e.decrypt(keys);
    fhe::Matrix MxN_d = MxN_e.decrypt(keys);
    
    std::cout << std::string(M_d) << std::endl;
    std::cout << std::string(N_d) << std::endl;
    std::cout << std::string(MpN_d) << std::endl;
    std::cout << std::string(MxN_d) << std::endl;

    return 0;

=======
    fhe::Matrix *M = new fhe::Matrix(2,2);
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
    ml::Network *net = new ml::Network(784,400,10,0.001);
    net->randomize_weights();
    std::string line;
    for(int epochs = 0; epochs < 20; epochs++){
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
    }
    std::ifstream test_file("../mnist_test.csv");
    int correct = 0;
    int total = 0;
    while(getline(test_file, line)) {
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
        fhe::Matrix *prediction = net->predict(input);
        std::cout << prediction->toString() << std::endl;
        int max = 0;
        for(int i = 0; i < 10; i++)
            if(prediction->at(i,0) > prediction->at(max,0))
                max = i;
        std::cout << "predicted:"+std::to_string(max)+" | target:"+std::to_string(target) <<std::endl;
        total++;
        if(max == target)
            correct++;
    }
    double percent = correct * 100. / total;
    std::cout << percent <<std::endl;
    return 0;
>>>>>>> 77aed8b (02.05.22)
}
