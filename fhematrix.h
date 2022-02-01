#ifndef _FHEMATRIX_H_
#define _FHEMATRIX_H_
namespace fhe {
    class Matrix {
        public:
            Matrix(int, int);

            int get_rows();
            int get_cols();

            Matrix operator + (Matrix);
            Matrix operator - (Matrix);
            Matrix operator * (Matrix);
            

        private:
            int rows;
            int cols;
            double **mat;
    };
}
#endif
