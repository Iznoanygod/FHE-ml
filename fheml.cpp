#include "fheml.h"
#include <math.h>
 #include <numeric>

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

fhe::FHEMatrix *sigmoid(const fhe::FHEMatrix *M) {
    fhe::FHEMatrix *temp = new fhe::FHEMatrix(M->get_rows(), M->get_cols(), M->get_cc());
    
    for(int i = 0; i < temp->get_rows(); i++) {
        for(int j = 0; j < temp->get_cols(); j++) {
            Ciphertext_t ct = M->at(i,j);
            Ciphertext_t sig = M->get_cc()->EvalPoly(ct, {0.000186759, 0.0, -0.0098156, 0.0, 0.22769, 0.5});
            temp->set(i,j,sig);
        }
    }

    return temp;
}

fhe::FHEMatrix *dsigmoid(const fhe::FHEMatrix *M) {
    fhe::FHEMatrix *temp = new fhe::FHEMatrix(M->get_rows(), M->get_cols(), M->get_cc());

    for(int i = 0; i < temp->get_rows(); i++) {
        for(int j = 0; j < temp->get_cols(); j++) {
            CryptoContext<DCRTPoly> cc = M->get_cc();
            Ciphertext_t ct = M->at(i,j);

            Ciphertext_t dsig = cc->EvalPoly(ct, {0.000933795, 0.0, -0.0294468, 0.0, 0.227692});
            
        }
    }

    return temp;
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
        this->bias_h = new fhe::FHEMatrix(h, 1, cc);
        this->bias_o = new fhe::FHEMatrix(o, 1, cc);
        this->l_rate = l_rate;
    }

    FHENetwork::~FHENetwork() {
        delete this->weights_ih;
        delete this->weights_ho;
        delete this->bias_h;
        delete this->bias_o;
    }

    fhe::FHEMatrix *FHENetwork::predict(fhe::FHEMatrix *input) {
        fhe::FHEMatrix *hidden = weights_ih->multiply(input);
        hidden->add(bias_h);
        fhe::FHEMatrix *hidden_sigmoid = sigmoid(hidden);

        fhe::FHEMatrix *output = weights_ho->multiply(hidden_sigmoid);
        output->add(bias_o);
        fhe::FHEMatrix *output_sigmoid = sigmoid(output);

        delete hidden;
        delete hidden_sigmoid;
        delete output;
        return output_sigmoid;
    }

    void FHENetwork::full_train(fhe::FHEMatrix *, fhe::FHEMatrix *) {

    }
    
    void FHENetwork::load_weights(fhe::FHEMatrix *ih, fhe::FHEMatrix *ho,
            fhe::FHEMatrix *bh, fhe::FHEMatrix *bo) {
        weights_ih = ih;
        weights_ho = ho;
        bias_h = bh;
        bias_o = bo;
    }

}
