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
            Matrix(int, int, int);

            int get_rows() const;
            int get_cols() const;
            int get_batch() const;

            Matrix operator + (Matrix) const;
            Matrix operator - (Matrix) const;
            Matrix operator * (Matrix) const;
            Matrix operator ^ (Matrix) const;
            Matrix operator ! () const;
            
            vector<double> *operator[] (int);
            operator std::string() const;
        private:
            int rows;
            int cols;
            int batch;
            vector<double> **mat;
    };

    class FHEMatrix {
        public:
            FHEMatrix(Matrix, CryptoContext<DCRTPoly>, Key_t);
            FHEMatrix(int, int, int, CryptoContext<DCRTPoly>);
            int get_rows();
            int get_cols();
            int get_batch();

            FHEMatrix operator + (FHEMatrix);
            FHEMatrix operator - (FHEMatrix);
            FHEMatrix operator * (FHEMatrix);
            FHEMatrix operator ^ (FHEMatrix);
            FHEMatrix operator ! ();

            Ciphertext_t *operator[] (int);
        
            Matrix decrypt(Key_t) const;
        private:
            int rows;
            int cols;
            int batch;
            CryptoContext<DCRTPoly> cc;
            Ciphertext_t **mat;
    };
}
#endif
