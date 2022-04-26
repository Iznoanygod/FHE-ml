#include "fheml.h"
#include <math.h>
#include <cstdio>

vector<double> sigmoid(vector<double> vec) {
    vector<double> v(vec.size());
    for(int i = 0; i < (int) vec.size(); i++) {
        v[i] = 1 / (1 + exp(-vec[i]));
    }
    return v;
}

vector<double> dsigmoid(vector<double> vec) {
    vector<double> v(vec.size());
    for(int i = 0; i < (int) vec.size(); i++) {
        double x = 1 / (1 + exp(-vec[i]));
        v[i] = x * (1 - x);
    }
    return v;
}

vector<double> add(vector<double> a, vector<double> b) {
    vector<double> v(a.size());
    for(int i = 0; i < (int) a.size(); i++) {
        v[i] = a[i] + b[i];
    }
    return v;
}

vector<double> subtract(vector<double> a, vector<double> b) {
    vector<double> v(a.size());
    for(int i = 0; i < (int) a.size(); i++) {
        v[i] = a[i] - b[i];
    }
    return v;
}

vector<double> multiply(vector<double> a, vector<double> b) {
    vector<double> v(a.size());
    for(int i = 0; i < (int) a.size(); i++) {
        v[i] = a[i] * b[i];
    }
    return v;
}

vector<double> multiply(vector<double> a, double val) {
    vector<double> v(a.size());
    for(int i = 0; i < (int) a.size(); i++) {
        v[i] = a[i] * val;
    }
    return v;
}

mat::Matrix *product(vector<double> a, vector<double> b) {
    int row = a.size();
    int col = b.size();

    mat::Matrix *mat = new mat::Matrix(row, col);
    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            mat->set(i, j, a[i] * b[j]);
    return mat;

}

namespace ml {
    Network::Network(int input, int hidden, int output, double l_rate) {
        this->l_rate = l_rate;

        this->weights_ih = new mat::Matrix(hidden, input);
        this->weights_ho = new mat::Matrix(output, hidden);
        this->bias_h.resize(hidden);
        this->bias_o.resize(output);
    }

    Network::~Network() {
        delete this->weights_ih;
        delete this->weights_ho;
    }

    vector<double> Network::predict(vector<double> input) const {
        vector<double> hidden = weights_ih->multiply(input);
        hidden = add(bias_h, hidden);
        vector<double> hidden_sigmoid = sigmoid(hidden);
        
        vector<double> output = weights_ho->multiply(hidden_sigmoid);
        output = add(bias_o, output);
        vector<double> output_sigmoid = sigmoid(output);
        
        return output_sigmoid;
    }

    void Network::train(vector<double> input, vector<double> target) {
        vector<double> hidden = weights_ih->multiply(input);
        hidden = add(bias_h, hidden);
        vector<double> hidden_sigmoid = sigmoid(hidden);
        
        vector<double> output = weights_ho->multiply(hidden_sigmoid);
        output = add(bias_o, output);
        vector<double> output_sigmoid = sigmoid(output);
        
        vector<double> error;
        for(int i = 0; i < (int) target.size(); i++)
            error.push_back(target[i]);
        
        error = subtract(error, output_sigmoid);
        vector<double> gradient = dsigmoid(output_sigmoid);
        gradient = multiply(gradient, error);
        gradient = multiply(gradient, l_rate);
        
        mat::Matrix *who_delta = product(gradient, hidden);
        weights_ho->add(who_delta);
        bias_o = add(bias_o, gradient);
        
        mat::Matrix *who_T = weights_ho->T();
        vector<double> hidden_errors = who_T->multiply(error);
        vector<double> h_gradient = dsigmoid(hidden);
        h_gradient = multiply(h_gradient, hidden_errors);
        h_gradient = multiply(h_gradient, l_rate);

        mat::Matrix *wih_delta = product(h_gradient, input);

        weights_ih->add(wih_delta);
        bias_h = add(bias_h, h_gradient);

        delete who_delta;
        delete who_T;
        delete wih_delta;
    }

    void Network::randomize_weights() {
        srand(time(NULL));
        for(int i = 0; i < weights_ih->get_rows(); i++)
            for(int j = 0; j < weights_ih->get_cols(); j++)
                weights_ih->set(i,j,((double)rand()/(double)RAND_MAX) * 2 - 1);

        for(int i = 0; i < weights_ho->get_rows(); i++)
            for(int j = 0; j < weights_ho->get_cols(); j++)
                weights_ho->set(i,j,((double)rand()/(double)RAND_MAX) * 2 - 1);

        for(int i = 0; i < (int) bias_h.size(); i++)
            bias_h[i] = ((double)rand()/(double)RAND_MAX) * 2 - 1;

        for(int i = 0; i < (int) bias_o.size(); i++)
            bias_o[i] = ((double)rand()/(double)RAND_MAX) * 2 - 1;
    }

    void Network::save(std::string file_path) const {
        FILE *fp = fopen(file_path.c_str(), "w");

        int input_size = weights_ih->get_cols();
        int hidden_size = weights_ih->get_rows();
        int output_size = weights_ho->get_rows();

        fwrite(&input_size, sizeof(input_size), 1, fp);
        fwrite(&hidden_size, sizeof(hidden_size), 1, fp);
        fwrite(&output_size, sizeof(output_size), 1, fp);
        fwrite(&l_rate, sizeof(l_rate), 1, fp);
        
        vector<vector<double>> ih_mat = weights_ih->get_mat();
        vector<vector<double>> ho_mat = weights_ho->get_mat();
        for(int i = 0; i < weights_ih->get_rows(); i++)
            for(int j = 0; j < weights_ih->get_cols(); j++)
                fwrite(&ih_mat[i][j], sizeof(double), 1, fp);

        for(int i = 0; i < weights_ho->get_rows(); i++)
            for(int j = 0; j < weights_ho->get_cols(); j++)
                fwrite(&ho_mat[i][j], sizeof(double), 1, fp);

        for(int i = 0; i < hidden_size; i++)
            fwrite(&bias_h[i], sizeof(double), 1, fp);
        
        for(int i = 0; i < output_size; i++)
            fwrite(&bias_o[i], sizeof(double), 1, fp);
        
        fclose(fp);
    }

    void Network::load(std::string file_path) {
        FILE *fp = fopen(file_path.c_str(), "r");
        if(fp == NULL) {
            std::cout << "Error: file doesn't exist" << std::endl;
            return;
        }

        int input_size;
        int hidden_size;
        int output_size;

        fread(&input_size, sizeof(input_size), 1, fp);
        fread(&hidden_size, sizeof(hidden_size), 1, fp);
        fread(&output_size, sizeof(output_size), 1, fp);
        fread(&l_rate, sizeof(l_rate), 1, fp);

        vector<vector<double>> ih_mat = weights_ih->get_mat();
        for(int i = 0; i < weights_ih->get_rows(); i++) {
            for(int j = 0; j < weights_ih->get_cols(); j++) {
                double in;
                fread(&in, sizeof(double), 1, fp);
                weights_ih->set(i, j, in);
            }
        }

        vector<vector<double>> ho_mat = weights_ho->get_mat();
        for(int i = 0; i < weights_ho->get_rows(); i++) {
            for(int j = 0; j < weights_ho->get_cols(); j++) {
                double in;
                fread(&in, sizeof(double), 1, fp);
                weights_ho->set(i, j, in);
            }
        }
        for(int i = 0; i < hidden_size; i++)
            fread(&bias_h[i], sizeof(double), 1, fp);
        for(int i = 0; i < output_size; i++)
            fread(&bias_o[i], sizeof(double), 1, fp);
        fclose(fp);
    }

    mat::Matrix *Network::get_weights_ih() const {
        return weights_ih;
    }

    mat::Matrix *Network::get_weights_ho() const {
        return weights_ho;
    }

    vector<double> Network::get_bias_h() const {
        return bias_h;
    }

    vector<double> Network::get_bias_o() const {
        return bias_o;
    }

    double Network::get_l_rate() const {
        return l_rate;
    }
    
    FHENetwork::FHENetwork(Network *net) {
        
        uint32_t multDepth = 10;
        uint32_t scaleFactorBits = 50;
        uint32_t batchSize = 784;
        SecurityLevel securityLevel = HEStd_128_classic;

        this->cc =
            CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
                    multDepth, scaleFactorBits, batchSize, securityLevel, 0, APPROXAUTO);
        
        /*usint m = 8192;
        usint init_size = 10;
        usint dcrtBits = 50;
        this->cc =
            CryptoContextFactory<DCRTPoly>::genCryptoContextCKKSWithParamsGen(
                m, init_size,
                dcrtBits, 10,
                784,
                OPTIMIZED, 20, 10,
                FIRSTMODSIZE, BV, APPROXAUTO);
        */
        this->cc->Enable(ENCRYPTION);
        this->cc->Enable(SHE);
        this->cc->Enable(LEVELEDSHE);
        
        this->key = this->cc->KeyGen();

        this->cc->EvalMultKeyGen(this->key.secretKey);
        this->cc->EvalSumKeyGen(this->key.secretKey);

        vector<int> rotations(200);
        for(int i = 1; i < 201; i++)
            rotations[i-1] = -i;
        this->cc->EvalAtIndexKeyGen(this->key.secretKey, rotations);

        this->weights_ih = new mat::FHEMatrix(net->get_weights_ih(), this->cc, this->key);
        this->weights_ho = new mat::FHEMatrix(net->get_weights_ho(), this->cc, this->key);

        Plaintext pbias_h = cc->MakeCKKSPackedPlaintext(net->get_bias_h());
        Plaintext pbias_o = cc->MakeCKKSPackedPlaintext(net->get_bias_o());
        
        this->bias_h = cc->Encrypt(this->key.publicKey, pbias_h);
        this->bias_o = cc->Encrypt(this->key.publicKey, pbias_o);

        this->l_rate = net->get_l_rate();
    }

    FHENetwork::~FHENetwork() {
        delete this->weights_ih;
        delete this->weights_ho;
    }

    Ciphertext<DCRTPoly> FHENetwork::first_predict(Ciphertext<DCRTPoly> input) const {
        Ciphertext<DCRTPoly> hidden = weights_ih->multiply(input);
        Ciphertext<DCRTPoly> hidden_bias = cc->EvalAdd(bias_h, hidden);
        //Ciphertext<DCRTPoly> hidden_sigmoid = cc->EvalPoly(hidden_bias, {0.5, 0.164128, 0, -0.00260371, 0, 0.000014906});
        std::cout << "checkpoint 1" << std::endl;
        /*std::cout << "checkpoint 2" << std::endl;
        Ciphertext<DCRTPoly> output = weights_ho->multiply(hidden_sigmoid);
        output = cc->EvalAdd(bias_o, output);
        std::cout << "checkpoint 3" << std::endl;
        Ciphertext<DCRTPoly> output_sigmoid = cc->EvalPoly(output, {0.5, 0.164128, 0, -0.00260371, 0, 0.000014906});
        return output_sigmoid;*/
        return hidden_bias;
    }

    Ciphertext<DCRTPoly> FHENetwork::second_predict(Ciphertext<DCRTPoly> input) const {
        Ciphertext<DCRTPoly> hidden = weights_ho->multiply(input);
        auto hidden_bias = cc->EvalAdd(bias_o, hidden);
        //Ciphertext<DCRTPoly> hidden_sigmoid = cc->EvalPoly(hidden_bias, {0.5, 0.164128, 0, -0.00260371, 0, 0.000014906});
        return hidden_bias;
    }
    
    CryptoContext<DCRTPoly> FHENetwork::get_cc() {
        return cc;
    }

    LPKeyPair<DCRTPoly> FHENetwork::get_key() {
        return key;
    }
}

