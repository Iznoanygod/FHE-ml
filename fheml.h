#ifndef _FHEML_H_
#define _FHEML_H_

#include "fhematrix.h"

namespace ml {
    class Network {
        public:
            Network(int, int, int, int, double);
            fhe::Matrix predict(fhe::Matrix);
        private:
            fhe::Matrix weights_ih;
            fhe::Matrix weights_ho;
            fhe::Matrix bias_h;
            fhe::Matrix bias_o;
            double l_rate;
    };
    class FHENetwork {
        private:
            fhe::FHEMatrix weights_ih;
            fhe::FHEMatrix weights_oh;
            fhe::FHEMatrix bias_h;
            fhe::FHEMatrix bias_o;
            double l_rate;
    };
}
#endif
