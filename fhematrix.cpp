#include "fhematrix.h"

namespace fhe {
    Matrix::Matrix(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;

        this->mat = new double*[rows];
        for(int i = 0; i < rows; i++)
            this->mat[i] = new double[cols];
    }

    int Matrix::get_rows() const {
        return rows;
    }

    int Matrix::get_cols() const {
        return cols;
    }

    Matrix Matrix::operator+ (Matrix M) const {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }
        
        Matrix temp(rows, cols);
        
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = mat[i][j] + M[i][j];

        return temp;
    }
    
    Matrix Matrix::operator- (Matrix M) const {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        Matrix temp(rows, cols);

        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = mat[i][j] - M[i][j];

        return temp;
    }
#ifdef __AVX2__
    Matrix Matrix::operator* (Matrix M) const {
        if(cols != M.get_rows()){
            throw -1;
        }

        Matrix temp(rows, M.get_cols());

        /*
         * TODO:
         *  - Write AVX implementation
         */

        return temp;
    }
#else
    Matrix Matrix::operator* (Matrix M) const {
        if(cols != M.get_rows()){
            throw -1;
        }
        
        Matrix temp(rows, M.get_cols());
        
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < M.get_cols(); j++){
                double sum = 0;
                for(int k = 0; k < cols; k++){
                    sum += mat[i][k] * M[k][j];
                }
                temp[i][j] = sum;
            }
        }

        return temp;
    }
#endif

    Matrix Matrix::operator^ (Matrix M) const {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        Matrix temp(rows, cols);
        
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = mat[i][j] * M[i][j];

        return temp;
    }

    Matrix Matrix::operator! () const {
        Matrix temp(cols, rows);

        for(int i = 0; i < cols; i++)
            for(int j = 0; j < rows; j++)
                temp[i][j] = mat[j][i];
    
        return temp;
    }

    double *Matrix::operator[] (int m){
        if(m < 0 || m >= rows)
            throw -1;

        return mat[m];
    }

    Matrix::operator std::string() const {
        std::string val = "{";

        return val;
    }

    FHEMatrix::FHEMatrix(int rows, int cols, int batch,
            Matrix mat, CryptoContext<DCRTPoly> cc, Key_t key) {
        this->rows = rows;
        this->cols = cols;
        this->batch = batch;
        this->cc = cc;

        this->mat = new Ciphertext_t*[rows];
        for(int i = 0; i < rows; i++) {
            this->mat[i] = new Ciphertext_t[cols];
            for(int j = 0; j < cols; j++){
                vector<double> v = {mat[i][j]};
                Plaintext ptxt = cc->MakeCKKSPackedPlaintext(v);
                this->mat[i][j] = cc->Encrypt(key.publicKey, ptxt);
            }
        }
        
    }

    FHEMatrix::FHEMatrix(int rows, int cols, int batch, CryptoContext<DCRTPoly> cc) {
        this->rows = rows;
        this->cols = cols;
        this->batch = batch;
        this->cc = cc;

        this->mat = new Ciphertext_t*[rows];
        for(int i = 0; i < rows; i++)
            this->mat[i] = new Ciphertext_t[cols];

    }

    int FHEMatrix::get_rows() {
        return rows;
    }
    
    int FHEMatrix::get_cols() {
        return cols;
    }

    int FHEMatrix::get_batch() {
        return batch;
    }

    FHEMatrix FHEMatrix::operator+ (FHEMatrix M) {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        FHEMatrix temp(rows, cols, batch, cc);
        
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = cc->EvalAdd(mat[i][j], M[i][j]);

        return temp;
    }

    FHEMatrix FHEMatrix::operator- (FHEMatrix M) {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        FHEMatrix temp(rows, cols, batch, cc);

        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = cc->EvalSub(mat[i][j], M[i][j]);

        return temp;
    }

    FHEMatrix FHEMatrix::operator* (FHEMatrix M) {
        if(cols != M.get_rows() || batch != M.get_batch()){
            throw -1;
        }    

        FHEMatrix temp(rows, M.get_cols(), batch, cc);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < M.get_cols(); j++) {
                Ciphertext_t sum = cc->EvalMult(mat[i][0], M[0][j]);
                for(int k = 1; k < cols; k++){
                    Ciphertext_t mul = cc->EvalMult(mat[i][k], M[k][j]);
                    sum = cc->EvalAdd(sum,mul);
                }
                temp[i][j] = sum;
            }
        }

        return temp;
    }

    Ciphertext_t *FHEMatrix::operator[] (int m) {
        if(m < 0 || m >= rows)
            throw -1;

        return mat[m];
    }
}

