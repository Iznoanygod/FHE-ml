#include "fheml.h"
#include <math.h>
 #include <numeric>

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
}

namespace ml {
    
    Network::Network(int i, int h, int o, int batch_size, double l_rate):
    weights_ih(fhe::Matrix(h, i, batch_size)),
    weights_ho(fhe::Matrix(o, h, batch_size)),
    bias_h(fhe::Matrix(h, 1, batch_size)),
    bias_o(fhe::Matrix(o, 1, batch_size))
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

    }
}
