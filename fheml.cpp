#include "fheml.h"
#include <math.h>

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
}
