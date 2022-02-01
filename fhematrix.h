#ifndef _FHEMATRIX_H_
#define _FHEMATRIX_H_

#ifdef __AVX2__
#include <x86intrin.h>
#endif

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
            
            double* operator[] (int);
        private:
            int rows;
            int cols;
            double **mat;
    };
}
#endif
