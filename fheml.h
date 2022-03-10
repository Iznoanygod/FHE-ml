#ifndef _FHEML_H_
#define _FHEML_H_

#include "fhematrix.h"

#define VECTOR_LENGTH 32

namespace ml {
    class Network {
        public:
            Network(int, int, int, double);
            ~Network();
            fhe::Matrix *predict(fhe::Matrix *);
            fhe::FHEMatrix *predict(fhe::FHEMatrix *);
            void train(fhe::Matrix *, fhe::Matrix *);
            void randomize_weights();
            void save(std::string);
            void load(std::string);
            fhe::Matrix *get_weights_ih();
            fhe::Matrix *get_weights_ho();
            fhe::Matrix *get_bias_h();
            fhe::Matrix *get_bias_o();
        private:
            fhe::Matrix *weights_ih;
            fhe::Matrix *weights_ho;
            fhe::Matrix *bias_h;
            fhe::Matrix *bias_o;
            double l_rate;
    };
    class FHENetwork {
        public:
            FHENetwork(int, int, int, double, CryptoContext<DCRTPoly>);
            FHENetwork(Network, CryptoContext<DCRTPoly>);
            ~FHENetwork();
            Ciphertext<DCRTPoly> predict(Ciphertext<DCRTPoly>);
            void full_train(fhe::FHEMatrix *, fhe::FHEMatrix *);
            void load_weights(fhe::FHEMatrix *, fhe::FHEMatrix*,
                    Ciphertext<DCRTPoly>, Ciphertext<DCRTPoly>);
        private:
            fhe::FHEMatrix *weights_ih;
            fhe::FHEMatrix *weights_ho;
            Ciphertext<DCRTPoly> bias_h;
            Ciphertext<DCRTPoly> bias_o;
            CryptoContext<DCRTPoly> cc;
            double l_rate;
    };
}
#endif
