#ifndef _FHEMATRIX_H_
#define _FHEMATRIX_H_

#include "palisade.h"

#ifdef __AVX2__
#include <x86intrin.h>
#endif

using namespace lbcrypto;

#define Ciphertext_t Ciphertext<DCRTPolyImpl<bigintfxd::BigVectorImpl<bigintfxd::BigInteger<unsigned int, BigIntegerBitLength>>>>
#define Key_t LPKeyPair<DCRTPolyImpl<bigintfxd::BigVectorImpl<bigintfxd::BigInteger<unsigned int, BigIntegerBitLength>>>>

namespace fhe {
    class Matrix {
        public:
            Matrix(int, int);
            ~Matrix();
            int get_rows() const;
            int get_cols() const;

            void add(double);
            void add(Matrix *);
            void subtract(double);
            void subtract(Matrix *);
            Matrix *T();
            void multiply(double);
            void element_multiply(Matrix *);
            Matrix *multiply(Matrix *) const;

            double at(int, int) const;
            void set(int, int, double);
            
            double **get_mat();
            Matrix *copy();
            std::string toString() const;
        private:
            int rows;
            int cols;
            double **mat;
    };

    class FHEMatrix {
        public:
            FHEMatrix(Matrix *, CryptoContext<DCRTPoly>, Key_t);
            FHEMatrix(int, int, CryptoContext<DCRTPoly>);
            ~FHEMatrix();
            int get_rows() const;
            int get_cols() const;
            CryptoContext<DCRTPoly> get_cc() const;

            void add(double);
            void add(FHEMatrix *);
            void subtract(double);
            void subtract(FHEMatrix *);
            FHEMatrix *T();
            void multiply(double);
            void element_multiply(FHEMatrix *);
            FHEMatrix *multiply(FHEMatrix *) const;

            Ciphertext_t at(int, int) const;
            void set(int, int, Ciphertext_t);
        
            Matrix *decrypt(Key_t) const;
        private:
            int rows;
            int cols;
            CryptoContext<DCRTPoly> cc;
            Ciphertext_t **mat;
    };
}
#endif
