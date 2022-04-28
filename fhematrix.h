#ifndef _FHEMATRIX_H_
#define _FHEMATRIX_H_

#include "palisade.h"

using namespace lbcrypto;

namespace mat {
    class Matrix{ 
        public:
            Matrix(int rows, int cols);
            int get_rows() const;
            int get_cols() const;
            vector<vector<double>> get_mat();

            void add(double val);
            void add(Matrix *M);

            void subtract(double val);
            void subtract(Matrix *M);

            Matrix *T();
            void multiply(double val);
            void element_multiply(Matrix *M);
            Matrix *multiply(Matrix *M) const;
            vector<double> multiply(vector<double> vec) const;

            double at(int row, int col) const;
            void set(int row, int col, double val);

            Matrix *copy() const;
            std::string toString() const;
        private:
            int rows;
            int cols;
            vector<vector<double>> mat;
    };

    class FHEMatrix {
        public:
            FHEMatrix(Matrix *mat, CryptoContext<DCRTPoly> cc, LPKeyPair<DCRTPoly> keys);
            FHEMatrix(int rows, int cols, vector<Ciphertext<DCRTPoly>> mat,
                    CryptoContext<DCRTPoly> cc);
            int get_rows() const;
            int get_cols() const;
            vector<Ciphertext<DCRTPoly>> get_mat() const;

            void add(double val);
            void add(FHEMatrix *M);

            void subtract(double val);
            void subtract(FHEMatrix *M);

            void multiply(double val);
            void element_multiply(FHEMatrix *M);
            FHEMatrix *multiply(FHEMatrix *M) const;
            Ciphertext<DCRTPoly> multiply(Ciphertext<DCRTPoly> vec) const;

            Ciphertext<DCRTPoly> at(int row) const;
            void set(int row, Ciphertext<DCRTPoly> vec);

            Matrix *decrypt(LPKeyPair<DCRTPoly> keys) const;

        private:
            int rows;
            int cols;
            CryptoContext<DCRTPoly> cc;
            vector<Ciphertext<DCRTPoly>> mat;
    };
}
#endif
