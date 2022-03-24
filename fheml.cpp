#include "fheml.h"
#include <math.h>
#include <numeric>
#include <cstdio>

fhe::Matrix *sigmoid(const fhe::Matrix *M) {
    fhe::Matrix *temp = new fhe::Matrix(M->get_rows(), M->get_cols());
    
    for(int i = 0; i < temp->get_rows(); i++) {
        for(int j = 0; j < temp->get_cols(); j++) {
            double sig = 1. / (1 + exp(-M->at(i,j)));
            temp->set(i,j,sig);
        }
    }
    
    return temp;
}

fhe::Matrix *dsigmoid(const fhe::Matrix *M) {
    fhe::Matrix *sig = sigmoid(M);
    
    for(int i = 0; i < sig->get_rows(); i++){
        for(int j = 0; j < sig->get_cols(); j++){
            double s = sig->at(i,j);
            sig->set(i,j,s * (1.0 - s));
        }
    }
    
    return sig;
}

Ciphertext<DCRTPoly> sigmoid(const Ciphertext<DCRTPoly> v, CryptoContext<DCRTPoly> cc) {
    Ciphertext<DCRTPoly> sig = cc->EvalPoly(v, {0.000186759, 0.0, -0.0098156, 0.0, 0.22769, 0.5});
    //auto sig = cc->EvalMult(v5, 0.000186759) + cc->EvalMult(v3, -0.0098156) + cc->EvalMult(v, 0.22769) + 0.5;
    return sig;
}

Ciphertext<DCRTPoly> dsigmoid(const Ciphertext<DCRTPoly> v, CryptoContext<DCRTPoly> cc) {
    Ciphertext<DCRTPoly> dsig = cc->EvalPoly(v, {0.000933795, 0.0, -0.0294468, 0.0, 0.227692});
    //auto v2 = cc->EvalMult(v, v);
    //auto v4 = cc->EvalMult(v2, v2);
    return dsig;
}

namespace ml {
    
    Network::Network(int i, int h, int o, double l_rate) {
        this->weights_ih = new fhe::Matrix(h, i);
        this->weights_ho = new fhe::Matrix(o, h);
        this->bias_h = new fhe::Matrix(h, 1);
        this->bias_o = new fhe::Matrix(o, 1);
        this->l_rate = l_rate;
    }

    Network::~Network() {
        delete this->weights_ih;
        delete this->weights_ho;
        delete this->bias_h;
        delete this->bias_o;
    }

    fhe::Matrix *Network::predict(fhe::Matrix *input) {
        fhe::Matrix *hidden = weights_ih->multiply(input);
        hidden->add(bias_h);
        fhe::Matrix *hidden_sigmoid = sigmoid(hidden);

        fhe::Matrix *output = weights_ho->multiply(hidden_sigmoid);
        output->add(bias_o);
        fhe::Matrix *output_sigmoid = sigmoid(output);

        delete hidden;
        delete hidden_sigmoid;
        delete output;
        return output_sigmoid;
    }

    void Network::train(fhe::Matrix *input, fhe::Matrix *target) {
        fhe::Matrix *hidden = weights_ih->multiply(input);
        hidden->add(bias_h);
        fhe::Matrix *hidden_sigmoid = sigmoid(hidden);

        fhe::Matrix *output = weights_ho->multiply(hidden_sigmoid);
        output->add(bias_o);
        fhe::Matrix *output_sigmoid = sigmoid(output);

        fhe::Matrix *error = target->copy();
        error->subtract(output_sigmoid);
        fhe::Matrix *gradient = dsigmoid(output_sigmoid);
        gradient->element_multiply(error);
        gradient->multiply(l_rate);

        fhe::Matrix *hidden_T = hidden->T();
        fhe::Matrix *who_delta = gradient->multiply(hidden_T);

        weights_ho->add(who_delta);
        bias_o->add(gradient);

        fhe::Matrix *who_T = weights_ho->T();
        fhe::Matrix *hidden_errors = who_T->multiply(error);

        fhe::Matrix *h_gradient = dsigmoid(hidden);
        h_gradient->element_multiply(hidden_errors);
        h_gradient->multiply(l_rate);

        fhe::Matrix *i_T = input->T();
        fhe::Matrix *wih_delta = h_gradient->multiply(i_T);

        weights_ih->add(wih_delta);
        bias_h->add(h_gradient);

        delete hidden;
        delete hidden_sigmoid;
        delete output;
        delete output_sigmoid;
        delete error;
        delete gradient;
        delete hidden_T;
        delete who_delta;
        delete who_T;
        delete hidden_errors;
        delete h_gradient;
        delete i_T;
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

        for(int i = 0; i < bias_h->get_rows(); i++)
            for(int j = 0; j < bias_h->get_cols(); j++)
                bias_h->set(i,j,((double)rand()/(double)RAND_MAX) * 2 - 1);

        for(int i = 0; i < bias_o->get_rows(); i++)
            for(int j = 0; j < bias_o->get_cols(); j++)
                bias_o->set(i,j,((double)rand()/(double)RAND_MAX) * 2 - 1);
    }
    
    void Network::save(std::string file_name) {
        FILE *fp = fopen(file_name.c_str(), "w");
        int input_size = weights_ih->get_cols();
        int hidden_size = weights_ih->get_rows();
        int output_size = weights_ho->get_rows();
        fwrite(&input_size, sizeof(input_size), 1, fp);
        fwrite(&hidden_size, sizeof(hidden_size), 1, fp);
        fwrite(&output_size, sizeof(output_size), 1, fp);
        fwrite(&l_rate, sizeof(l_rate), 1, fp);
        double **ih_mat = weights_ih->get_mat();
        double **ho_mat = weights_ho->get_mat();
        for(int i = 0; i < weights_ih->get_rows(); i++) {
            fwrite(ih_mat[i], sizeof(*ih_mat[i]), weights_ih->get_cols(), fp);
        }
        for(int i = 0; i < weights_ho->get_rows(); i++) {
            fwrite(ho_mat[i], sizeof(*ho_mat[i]), weights_ho->get_cols(), fp);
        }
        double **bh_mat = bias_h->get_mat();
        double **bo_mat = bias_o->get_mat();
        for(int i = 0; i < bias_h->get_rows(); i++) {
            fwrite(bh_mat[i], sizeof(*bh_mat[i]), bias_h->get_cols(), fp);
        }
        for(int i = 0; i < bias_o->get_rows(); i++) {
            fwrite(bo_mat[i], sizeof(*bh_mat[i]), bias_o->get_cols(), fp);
        }

        fclose(fp);
    }

    void Network::load(std::string file_name) {
        FILE *fp = fopen(file_name.c_str(), "r");
        if(fp == NULL){
            std::cout << "Error: file doesn't exist" << std::endl;
            throw -1;
        }
        int input_size;
        int hidden_size;
        int output_size;
        double l_rate;

        fread(&input_size, sizeof(input_size), 1, fp);
        fread(&hidden_size, sizeof(hidden_size), 1, fp);
        fread(&output_size, sizeof(output_size), 1, fp);
        fread(&l_rate, sizeof(l_rate), 1, fp);
        double **ih_mat = weights_ih->get_mat();
        double **ho_mat = weights_ho->get_mat();
        for(int i = 0; i < weights_ih->get_rows(); i++) {
            fread(ih_mat[i], sizeof(*ih_mat[i]), weights_ih->get_cols(), fp);
        }
        for(int i = 0; i < weights_ho->get_rows(); i++) {
            fread(ho_mat[i], sizeof(*ho_mat[i]), weights_ho->get_cols(), fp);
        }
        double **bh_mat = bias_h->get_mat();
        double **bo_mat = bias_o->get_mat();
        for(int i = 0; i < bias_h->get_rows(); i++) {
            fread(bh_mat[i], sizeof(*bh_mat[i]), bias_h->get_cols(), fp);
        }
        for(int i = 0; i < bias_o->get_rows(); i++) {
            fread(bo_mat[i], sizeof(*bh_mat[i]), bias_o->get_cols(), fp);
        }
        fclose(fp);
    }

    fhe::Matrix *Network::get_weights_ih() {
        return weights_ih;
    }

    fhe::Matrix *Network::get_weights_ho() {
        return weights_ho;
    }

    fhe::Matrix *Network::get_bias_h() {
        return bias_h;
    }

    fhe::Matrix *Network::get_bias_o() {
        return bias_o;
    }

    FHENetwork::FHENetwork(int i, int h, int o, double l_rate,
            CryptoContext<DCRTPoly> cc) {
        this->weights_ih = new fhe::FHEMatrix(h, i, cc);
        this->weights_ho = new fhe::FHEMatrix(o, h, cc);
        //this->bias_h = new fhe::FHEMatrix(h, 1, cc);
        //this->bias_o = new fhe::FHEMatrix(o, 1, cc);
        this->l_rate = l_rate;
        this->cc = cc;
    }

    FHENetwork::FHENetwork(Network *n, CryptoContext<DCRTPoly> cc, Key_t key){
        this->weights_ih = new fhe::FHEMatrix(n->get_weights_ih(), cc, key);
        this->weights_ho = new fhe::FHEMatrix(n->get_weights_ho(), cc, key);
        
        double** mat_bh = n->get_bias_h()->get_mat();
        vector<double> bh_vector;
        bh_vector.resize(n->get_bias_h()->get_rows());
        for(int i = 0; i < n->get_bias_h()->get_rows(); i++){
            bh_vector[i] = mat_bh[i][0];
        }
        
        double** mat_bo = n->get_bias_o()->get_mat();
        vector<double> bo_vector;
        bo_vector.resize(n->get_bias_o()->get_rows());
        for(int i = 0; i < n->get_bias_o()->get_rows(); i++){
            bo_vector[i] = mat_bo[i][0];
        }

        Plaintext ptxt_bh = cc->MakeCKKSPackedPlaintext(bh_vector);
        Plaintext ptxt_bo = cc->MakeCKKSPackedPlaintext(bo_vector);

        this->bias_h = cc->EvalMult(cc->EvalMult(cc->Encrypt(key.publicKey, ptxt_bh), 1),1);
        this->bias_o = cc->EvalMult(cc->EvalMult(cc->Encrypt(key.publicKey, ptxt_bo), 1),1);

        this->l_rate = l_rate;
        this->cc = cc;
    }

    FHENetwork::~FHENetwork() {
        delete this->weights_ih;
        delete this->weights_ho;
        //delete this->bias_h;
        //delete this->bias_o;
    }

    Ciphertext<DCRTPoly> FHENetwork::predict_first_layer(Ciphertext<DCRTPoly> input) {
        Ciphertext<DCRTPoly> hidden = weights_ih->multiply(input);
        std::cout << "Multiplied input with weights" << std::endl;
        hidden = cc->EvalAdd(bias_h, hidden);
        std::cout << "Added biases" << std::endl;
        //Ciphertext<DCRTPoly> hidden_sigmoid = sigmoid(hidden, cc);
        //std::cout << "Performed sigmoid" << std::endl;
        //Ciphertext<DCRTPoly> output = weights_ho->multiply(hidden_sigmoid);
        //std::cout << "Multiplied with weights" << std::endl;
        //output = cc->EvalAdd(bias_o, hidden);
        //std::cout << "Added biases" << std::endl;
        //Ciphertext<DCRTPoly> output_sigmoid = sigmoid(output, cc);
        //std::cout << "Sigmoid" << std::endl;
        //delete hidden;
        //delete hidden_sigmoid;
        //delete output;
        return hidden;
    }

    Ciphertext<DCRTPoly> FHENetwork::predict_second_layer(Ciphertext<DCRTPoly> input) {
        Ciphertext<DCRTPoly> hidden = weights_ho->multiply(input);
        std::cout << "Multiplied by weights round 2" << std::endl;

        hidden = cc->EvalAdd(bias_o, hidden);
        std::cout << "Added second bias" << std::endl;
        return hidden;
    }

    void FHENetwork::full_train(fhe::FHEMatrix *, fhe::FHEMatrix *) {

    }
    
    void FHENetwork::load_weights(fhe::FHEMatrix *ih, fhe::FHEMatrix *ho,
            Ciphertext<DCRTPoly> bh, Ciphertext<DCRTPoly> bo) {
        delete this->weights_ih;
        delete this->weights_ho;

        weights_ih = ih;
        weights_ho = ho;
        bias_h = bh;
        bias_o = bo;
    }

}
