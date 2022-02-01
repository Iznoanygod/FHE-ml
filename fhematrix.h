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

            int get_rows();
            int get_cols();

            Matrix operator + (Matrix);
            Matrix operator - (Matrix);
            Matrix operator * (Matrix);
            Matrix operator ^ (Matrix);
            Matrix operator ! ();
            
            double *operator[] (int);
        private:
            int rows;
            int cols;
            double **mat;
    };

    class FHEMatrix {
        public:
            FHEMatrix(int, int, int, Matrix, CryptoContext<DCRTPoly>, Key_t);
            
            int get_rows();
            int get_cols();
            int get_batch();

            FHEMatrix operator + (FHEMatrix);
            FHEMatrix operator - (FHEMatrix);
            FHEMatrix operator * (FHEMatrix);
            FHEMatrix operator ^ (FHEMatrix);
            FHEMatrix operator ! ();

            Ciphertext_t *operator[] (int);
        private:
            FHEMatrix(int, int, int, CryptoContext<DCRTPoly>);
            int rows;
            int cols;
            int batch;
            CryptoContext<DCRTPoly> cc;
            Ciphertext_t **mat;
    };
}
#endif
