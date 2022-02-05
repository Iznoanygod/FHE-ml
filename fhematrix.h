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
            
            Matrix *copy();
            std::string toString() const;
        private:
            int rows;
            int cols;
            double **mat;
    };

    class FHEMatrix {
        public:
<<<<<<< HEAD
            FHEMatrix(Matrix, int, CryptoContext<DCRTPoly>, Key_t);
            FHEMatrix(int, int, int, CryptoContext<DCRTPoly>);
=======
            FHEMatrix(Matrix *, CryptoContext<DCRTPoly>, Key_t);
            FHEMatrix(int, int, CryptoContext<DCRTPoly>);
            ~FHEMatrix();
>>>>>>> 77aed8b (02.05.22)
            int get_rows();
            int get_cols();

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
        
<<<<<<< HEAD
            Matrix decrypt(Key_t) const;
=======
            Matrix *decrypt(Key_t) const;
>>>>>>> 77aed8b (02.05.22)
        private:
            int rows;
            int cols;
            CryptoContext<DCRTPoly> cc;
            Ciphertext_t **mat;
    };
}
#endif
