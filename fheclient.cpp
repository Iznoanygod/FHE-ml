#include "fheml.h"
#include <fstream>

int main(int argc, char** argv) {
    uint32_t multDepth = 1;
    uint32_t scaleFactorBits = 20;
    uint32_t batchSize = 1;

    SecurityLevel securityLevel = HEStd_128_classic;

	CryptoContext<DCRTPoly> cc =
		CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
			multDepth, scaleFactorBits, batchSize, securityLevel);

	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);

    auto keys = cc->KeyGen();
    
    cc->EvalMultKeyGen(keys.secretKey);
    std::cout << sizeof(Ciphertext_t) << std::endl;
    
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
    ml::Network *net = new ml::Network(10,10,10,0.001);
    net->randomize_weights();
    /*std::string line;
    for(int epochs = 0; epochs < 1; epochs++){
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
    */
    std::cout << "Creating FHE Network" << std::endl;
    ml::FHENetwork *fhenet = new ml::FHENetwork(10, 10, 10, 0.001, cc);
    std::cout << "Loading model weights" << std::endl;
    fhe::FHEMatrix *wih_e = new fhe::FHEMatrix(net->get_weights_ih(), cc, keys);
    std::cout << "Loaded first set of weights" << std::endl;
    wih_e->multiply(1.0);
    //fhe::FHEMatrix *who_e = new fhe::FHEMatrix(net->get_weights_ho(), cc, keys);
    //fhe::FHEMatrix *bh_e = new fhe::FHEMatrix(net->get_bias_h(), cc, keys);
    //fhe::FHEMatrix *bo_e = new fhe::FHEMatrix(net->get_bias_o(), cc, keys);
    //fhenet->load_weights(wih_e, who_e, bh_e, bo_e);
    
    fhenet->full_train(NULL, NULL);
    delete fhenet;
    delete net;
    delete wih_e;
    
    return 0;
}
