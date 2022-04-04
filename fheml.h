#ifndef _FHEML_H_
#define _FHEML_H_

#include "fhematrix.h"

namespace ml {
    class Network {
        public:
            Network(int input, int hidden, int output, double l_rate);
            ~Network();

            vector<double> predict(vector<double> input) const;
            void train(vector<double> input, vector<double> target);

            void randomize_weights();

            void save(std::string file_path) const;
            void load(std::string file_path);
            
            mat::Matrix *get_weights_ih() const;
            mat::Matrix *get_weights_ho() const;
            vector<double> get_bias_h() const;
            vector<double> get_bias_o() const;
            double get_l_rate() const;
        private:
            mat::Matrix *weights_ih;
            mat::Matrix *weights_ho;
            vector<double> bias_h;
            vector<double> bias_o;
            double l_rate;
    };

    class FHENetwork {
        public:
            FHENetwork(Network *net);
            ~FHENetwork();
            
            Ciphertext<DCRTPoly> predict(Ciphertext<DCRTPoly> input) const;
            
            CryptoContext<DCRTPoly> get_cc();
            LPKeyPair<DCRTPoly> get_key();
        private:
            mat::FHEMatrix *weights_ih;
            mat::FHEMatrix *weights_ho;
            
            Ciphertext<DCRTPoly> bias_h;
            Ciphertext<DCRTPoly> bias_o;

            double l_rate;

            CryptoContext<DCRTPoly> cc;
            LPKeyPair<DCRTPoly> key;
    };
}

#endif
