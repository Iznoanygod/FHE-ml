#include "fhematrix.h"

namespace fhe {
    Matrix::Matrix(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;

        this->mat = new double *[rows];
        for(int i = 0; i < rows; i++)
            this->mat[i] = new double[cols];
    }

    Matrix::~Matrix() {
        for(int i = 0; i < rows; i++)
            delete[] this->mat[i];
        delete[] mat;
    }
    
    int Matrix::get_rows() const {
        return rows;
    }

    int Matrix::get_cols() const {
        return cols;
    }

    void Matrix::add(double scalar) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] += scalar;
    }

    void Matrix::add(Matrix *M) {
        if(rows != M->get_rows() || cols != M->get_cols())
            throw -1;

        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] += M->at(i,j);
    }

    void Matrix::subtract(double scalar) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] -= scalar;
    }

    void Matrix::subtract(Matrix *M) {
        if(rows != M->get_rows() || cols != M->get_cols())
            throw -1;

        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] -= M->at(i,j);
    }

    Matrix *Matrix::T() {
        Matrix *transpose = new Matrix(cols, rows);
        for(int i = 0; i < cols; i++)
            for(int j = 0; j < rows; j++)
                transpose->set(i, j, mat[j][i]);

        return transpose;
    }

    void Matrix::multiply(double scalar) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] *= scalar;
    }

    void Matrix::element_multiply(Matrix *M) {
        if(rows != M->get_rows() || cols != M->get_cols())
            throw -1;

        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] *= M->at(i,j);
    }
#ifdef __AVX2__
#else
    Matrix *Matrix::multiply(Matrix *M) const{
        if(cols != M->get_rows())
            throw -1;

        Matrix *temp = new Matrix(rows, M->get_cols());
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < M->get_cols(); j++) {
                double sum = 0;
                for(int k = 0; k < cols; k++)
                    sum += mat[i][k] * M->at(k,j);
                temp->set(i,j,sum);
            }
        }
        return temp;
    }
#endif
    double Matrix::at(int m, int n) const {
        return mat[m][n];
    }

    void Matrix::set(int m, int n, double j) {
        mat[m][n] = j;
    }

    Matrix *Matrix::copy() {
        Matrix *temp = new Matrix(rows, cols);

        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp->set(i,j,mat[i][j]);

        return temp;
    }

    std::string Matrix::toString() const {
        std::string val = "{";
        for(int j = 0; j < rows; j++){
            val += "{";
            for(int k = 0; k < cols; k++){
                val += std::to_string(mat[j][k]);
                if(k != cols - 1)
                    val += ",";
            }
            val += "}";
            if(j != rows-1)
                val += ",\n";
        }
        val += "}";
        return val;
    }

    FHEMatrix::FHEMatrix(Matrix *mat, CryptoContext<DCRTPoly> cc, Key_t key) {
        this->rows = mat->get_rows();
        this->cols = mat->get_cols();
        this->cc = cc;
        this->mat = new Ciphertext_t*[rows];
        for(int i = 0; i < rows; i++) {
            this->mat[i] = new Ciphertext_t[cols];
            for(int j = 0; j < cols; j++){
                vector<double> v = {mat->at(i,j)};
                Plaintext ptxt = cc->MakeCKKSPackedPlaintext(v);
                this->mat[i][j] = cc->Encrypt(key.publicKey, ptxt);
                v.clear();
            }
        }
        
    }

    FHEMatrix::FHEMatrix(int rows, int cols, CryptoContext<DCRTPoly> cc) {
        this->rows = rows;
        this->cols = cols;
        this->cc = cc;

        this->mat = new Ciphertext_t*[rows];
        for(int i = 0; i < rows; i++)
            this->mat[i] = new Ciphertext_t[cols];

    }

    FHEMatrix::~FHEMatrix() {
        for(int i = 0 ; i < rows; i++)
            delete[] mat[i];
        delete[] mat;
    }

    int FHEMatrix::get_rows() const {
        return rows;
    }
    
    int FHEMatrix::get_cols() const {
        return cols;
    }

    CryptoContext<DCRTPoly> FHEMatrix::get_cc() const {
        return cc;
    }

    void FHEMatrix::add(double scalar) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] = cc->EvalAdd(mat[i][j], scalar);
    }

    void FHEMatrix::add(FHEMatrix *M) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] = cc->EvalAdd(mat[i][j], M->at(i,j));
    }

    void FHEMatrix::subtract(double scalar) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] = cc->EvalSub(mat[i][j], scalar);
    }

    void FHEMatrix::subtract(FHEMatrix *M) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] = cc->EvalSub(mat[i][j], M->at(i,j));
    }

    FHEMatrix *FHEMatrix::T() {
        FHEMatrix *transpose = new FHEMatrix(cols, rows, cc);
        
        for(int i = 0; i < cols; i++)
            for(int j = 0; j < rows; j++)
                transpose->set(i,j,mat[j][i]);

        return transpose;
    }

    void FHEMatrix::multiply(double scalar) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] = cc->EvalMult(mat[i][j], scalar);
    }

    void FHEMatrix::element_multiply(FHEMatrix *M) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < M->get_cols(); j++) {
                mat[i][j] = cc->EvalMult(mat[i][j], M->at(i,j));
            }
        }
    }

    FHEMatrix *FHEMatrix::multiply(FHEMatrix *M) const {
        if(cols != M->get_rows())
            throw -1;

        FHEMatrix *temp = new FHEMatrix(rows, M->get_cols(), cc);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < M->get_cols(); j++) {
                Ciphertext_t sum = cc->EvalMult(mat[i][0], M->at(0,j));
                for(int k = 1; k < cols; k++) {
                    Ciphertext_t val = cc->EvalMult(mat[i][k], M->at(k,j));
                    sum = cc->EvalAdd(sum, val);
                }
                temp->set(i,j,sum);
            }
        }

        return temp;
    }

    Ciphertext_t FHEMatrix::at(int m, int n) const {
        return mat[m][n];
    }

    void FHEMatrix::set(int m, int n, Ciphertext_t c) {
        mat[m][n] = c;
    }

    Matrix *FHEMatrix::decrypt(Key_t key) const {
        Matrix *temp = new Matrix(rows, cols);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                Plaintext result;
                cc->Decrypt(key.secretKey, mat[i][j], &result);
                result->SetLength(1);
                temp->set(i,j,result->GetRealPackedValue()[0]);
            }
        }
        return temp;
    }
}

