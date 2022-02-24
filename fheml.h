#ifndef _FHEML_H_
#define _FHEML_H_

#include "fhematrix.h"

namespace ml {
    class Network {
        public:
            Network(int, int, int, double);
            ~Network();
            fhe::Matrix *predict(fhe::Matrix *);
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
            ~FHENetwork();
            fhe::FHEMatrix *predict(fhe::FHEMatrix *);
            void full_train(fhe::FHEMatrix *, fhe::FHEMatrix *);
            void load_weights(fhe::FHEMatrix *, fhe::FHEMatrix*,
                    fhe::FHEMatrix *, fhe::FHEMatrix *);
        private:
            fhe::FHEMatrix *weights_ih;
            fhe::FHEMatrix *weights_ho;
            fhe::FHEMatrix *bias_h;
            fhe::FHEMatrix *bias_o;
            double l_rate;
    };
}
#endif
