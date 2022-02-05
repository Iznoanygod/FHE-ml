#ifndef _FHEML_H_
#define _FHEML_H_

#include "fhematrix.h"

namespace ml {
    class Network {
        public:
            Network(int, int, int, double);
<<<<<<< HEAD
            fhe::Matrix predict(fhe::Matrix);
            void train(fhe::Matrix, fhe::Matrix);
        private:
            fhe::Matrix weights_ih;
            fhe::Matrix weights_ho;
            fhe::Matrix bias_h;
            fhe::Matrix bias_o;
=======
            fhe::Matrix *predict(fhe::Matrix *);
            void train(fhe::Matrix *, fhe::Matrix *);
            void randomize_weights();
        private:
            fhe::Matrix *weights_ih;
            fhe::Matrix *weights_ho;
            fhe::Matrix *bias_h;
            fhe::Matrix *bias_o;
>>>>>>> 77aed8b (02.05.22)
            double l_rate;
    };
    class FHENetwork {
        private:
<<<<<<< HEAD
            fhe::FHEMatrix weights_ih;
            fhe::FHEMatrix weights_oh;
            fhe::FHEMatrix bias_h;
            fhe::FHEMatrix bias_o;
=======
            fhe::FHEMatrix *weights_ih;
            fhe::FHEMatrix *weights_oh;
            fhe::FHEMatrix *bias_h;
            fhe::FHEMatrix *bias_o;
>>>>>>> 77aed8b (02.05.22)
            double l_rate;
    };
}
#endif
