#include "fhematrix.h"

<<<<<<< HEAD
#define MAX(a,b) a > b ? a : b

vector<double> operator+(double A, vector<double> B) {
    vector<double> temp(B.size());

    for(unsigned int i = 0; i < B.size(); i++) {
        temp[i] = A + B[i];
    }

    return temp;
}

vector<double> operator*(double A, vector<double> B) {
    vector<double> temp(B.size());

    for(unsigned int i = 0; i < B.size(); i++) {
        temp[i] = A * B[i];
    }

    return temp;
}

vector<double> operator-(double A, vector<double> B) {
    vector<double> temp(B.size());

    for(unsigned int i = 0; i < B.size(); i++) {
        temp[i] = A - B[i];
    }
    
    return temp;
}

vector<double> operator+(vector<double> A, vector<double> B) {
    if(A.size() == 1)
        return A[0] + B;
    if(B.size() == 1)
        return B[0] + A;
    if(A.size() != B.size())
        throw -1;

    vector<double> C(A.size());

    for(unsigned int i = 0; i < A.size(); i++) {
        C[i] = A[i]+B[i];
    }

    return C;
}

vector<double> operator*(vector<double> A, vector<double> B) {
    if(A.size() == 1)
        return A[0] * B;
    if(B.size() == 1)
        return B[0] * A;
    if(A.size() != B.size())
        throw -1;

    vector<double> C(A.size());

    for(unsigned int i = 0; i < A.size(); i++){
        C[i] = A[i]*B[i];
    }

    return C;
}

vector<double> operator-(vector<double> A, vector<double> B) {
    if(A.size() == 1)
        return A[0] - B;
    if(B.size() == 1)
        return -B[0] + A;
    if(A.size() != B.size())
        throw -1;

    vector<double> C(A.size());

    for(unsigned int i = 0; i < A.size(); i++) {
        C[i] = A[i]-B[i];
    }

    return C;
}

vector<double> & operator+=(vector<double> &A, const vector<double> &B) {
    if(A.size() != B.size())
        throw -1;

    for(unsigned int i = 0; i < A.size(); i++) {
        A[i] += B[i];
    }
    return A;
}

=======
>>>>>>> 77aed8b (02.05.22)
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

<<<<<<< HEAD
    int Matrix::get_batch() const {
        return batch;
    }

    Matrix Matrix::operator+ (Matrix M) const {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        int b = MAX(batch, M.get_batch());
        Matrix temp(rows, cols, b);
        
=======
    void Matrix::add(double scalar) {
>>>>>>> 77aed8b (02.05.22)
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] += scalar;
    }
<<<<<<< HEAD
    
    Matrix Matrix::operator- (Matrix M) const {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }
        
        int b = MAX(batch, M.get_batch());
        Matrix temp(rows, cols, b);
=======

    void Matrix::add(Matrix *M) {
        if(rows != M->get_rows() || cols != M->get_cols())
            throw -1;
>>>>>>> 77aed8b (02.05.22)

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
<<<<<<< HEAD
        }

        int b = MAX(batch, M.get_batch());
        Matrix temp(rows, M.get_cols(), b);
        
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < M.get_cols(); j++){
                vector<double> sum(b);
                fill(sum.begin(), sum.end(), 0);
                for(int k = 0; k < cols; k++){
                    sum += mat[i][k] * M[k][j];
                }
                temp[i][j] = sum;
=======

        Matrix *temp = new Matrix(rows, M->get_cols());
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < M->get_cols(); j++) {
                double sum = 0;
                for(int k = 0; k < cols; k++)
                    sum += mat[i][k] * M->at(k,j);
                temp->set(i,j,sum);
>>>>>>> 77aed8b (02.05.22)
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

<<<<<<< HEAD
        int b = MAX(batch, M.get_batch());
        Matrix temp(rows, cols, b);
        
=======
>>>>>>> 77aed8b (02.05.22)
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

<<<<<<< HEAD
    FHEMatrix::FHEMatrix(Matrix mat, int batch, CryptoContext<DCRTPoly> cc, Key_t key) {
        this->rows = mat.get_rows();
        this->cols = mat.get_cols();
        this->batch = batch;
=======
    FHEMatrix::FHEMatrix(Matrix *mat, CryptoContext<DCRTPoly> cc, Key_t key) {
        this->rows = mat->get_rows();
        this->cols = mat->get_cols();
>>>>>>> 77aed8b (02.05.22)
        this->cc = cc;

        this->mat = new Ciphertext_t*[rows];
        for(int i = 0; i < rows; i++) {
            this->mat[i] = new Ciphertext_t[cols];
            for(int j = 0; j < cols; j++){
<<<<<<< HEAD
                
                if(mat[i][j].size() == 1) {
                    vector<double> v(batch);
                    std::fill(v.begin(), v.end(), mat[i][j][0]);
                    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(v);
                    this->mat[i][j] = cc->Encrypt(key.publicKey, ptxt);
                }
                else {
                    vector<double> v = mat[i][j];
                    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(v);
                    this->mat[i][j] = cc->Encrypt(key.publicKey, ptxt);
                }
=======
                vector<double> v = {mat->at(i,j)};
                Plaintext ptxt = cc->MakeCKKSPackedPlaintext(v);
                this->mat[i][j] = cc->Encrypt(key.publicKey, ptxt);
>>>>>>> 77aed8b (02.05.22)
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

    int FHEMatrix::get_rows() {
        return rows;
    }
    
    int FHEMatrix::get_cols() {
        return cols;
    }

<<<<<<< HEAD
    int FHEMatrix::get_batch() {
        return batch;
    }

    FHEMatrix FHEMatrix::operator+ (FHEMatrix M) {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        int b = MAX(batch, M.get_batch());
        FHEMatrix temp(rows, cols, b, cc);
        
=======
    void FHEMatrix::add(double scalar) {
>>>>>>> 77aed8b (02.05.22)
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

<<<<<<< HEAD
    Matrix FHEMatrix::decrypt(Key_t key) const {
        Matrix temp(rows, cols, batch);
=======
    Matrix *FHEMatrix::decrypt(Key_t key) const {
        Matrix *temp = new Matrix(rows, cols);
>>>>>>> 77aed8b (02.05.22)

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

