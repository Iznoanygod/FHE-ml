#include "fheml.h"
#include <math.h>
 #include <numeric>

<<<<<<< HEAD
fhe::Matrix sigmoid(fhe::Matrix M) {
    fhe::Matrix temp(M.get_rows(), M.get_cols(), M.get_batch());

    for(int i = 0; i < temp.get_rows(); i++) {
        for(int j = 0; j < temp.get_rows(); j++) {
            vector<double> sig(M.get_batch());

            for(int k = 0; k < M.get_batch(); k++){
                sig[k] = 1. / (1 + exp(-M[i][j][k]));
            }
            
            temp[i][j] = sig;
        }
    }
    
    return temp;
}

fhe::Matrix dsigmoid(fhe::Matrix M) {
    fhe::Matrix sig = sigmoid(M);
    fhe::Matrix ones(sig.get_rows(), sig.get_cols(), sig.get_batch());
    for(int i = 0; i < ones.get_rows(); i++){
        for(int j = 0; j < ones.get_cols(); j++){
            std::fill(ones[i][j].begin(), ones[i][j].end(), 1.0);
        }
    }
    return sig * (ones - sig);
=======
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
>>>>>>> 77aed8b (02.05.22)
}

namespace ml {
    
<<<<<<< HEAD
    Network::Network(int i, int h, int o, double l_rate):
    weights_ih(fhe::Matrix(h, i, 1)),
    weights_ho(fhe::Matrix(o, h, 1)),
    bias_h(fhe::Matrix(h, 1, 1)),
    bias_o(fhe::Matrix(o, 1, 1))
    {
        this->l_rate = l_rate;
    }

    fhe::Matrix Network::predict(fhe::Matrix input) {
        fhe::Matrix hidden = weights_ih * input;
        hidden = hidden + bias_h;
        hidden = sigmoid(hidden);

        fhe::Matrix output = weights_ho * hidden;
        output = output + bias_o;
        output = sigmoid(output);

        return output;
    }

    void Network::train(fhe::Matrix input, fhe::Matrix target) {
        fhe::Matrix output = predict(input);

        fhe::Matrix error_batch = target - output;

        fhe::Matrix error(error_batch.get_rows(), error_batch.get_cols(),
                error_batch.get_batch());
        for(int i = 0; i < error_batch.get_rows(); i++){
            for(int j = 0; j < error_batch.get_cols(); j++){
                double sum = std::accumulate(error_batch[i][j].begin(),
                        error_batch[i][j].end(), 0.0f);
                std::fill(error[i][j].begin(), error[i][j].end(), sum);
            }
        }

        fhe::Matrix gradient = dsigmoid(output);
        gradient = gradient ^ error;

=======
    Network::Network(int i, int h, int o, double l_rate) {
        this->weights_ih = new fhe::Matrix(h, i);
        this->weights_ho = new fhe::Matrix(o, h);
        this->bias_h = new fhe::Matrix(h, 1);
        this->bias_o = new fhe::Matrix(o, 1);
        this->l_rate = l_rate;
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
>>>>>>> 77aed8b (02.05.22)
    }
}
