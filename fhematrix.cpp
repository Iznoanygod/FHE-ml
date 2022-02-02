#include "fhematrix.h"

vector<double> operator+(vector<double> A, vector<double> B) {
    if(A.size() != B.size())
        throw -1;

    vector<double> C(A.size());

    for(unsigned int i = 0; i < A.size(); i++) {
        C[i] = A[i]+B[i];
    }

    return C;
}

vector<double> operator*(vector<double> A, vector<double> B) {
    if(A.size() != B.size())
        throw -1;

    vector<double> C(A.size());

    for(unsigned int i = 0; i < A.size(); i++){
        C[i] = A[i]*B[i];
    }

    return C;
}

vector<double> operator-(vector<double> A, vector<double> B) {
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

namespace fhe {
    Matrix::Matrix(int rows, int cols, int batch) {
        this->rows = rows;
        this->cols = cols;
        this->batch = batch;

        this->mat = new vector<double> *[rows];
        for(int i = 0; i < rows; i++)
            this->mat[i] = new vector<double>[cols];
    }

    int Matrix::get_rows() const {
        return rows;
    }

    int Matrix::get_cols() const {
        return cols;
    }

    int Matrix::get_batch() const {
        return batch;
    }

    Matrix Matrix::operator+ (Matrix M) const {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }
        
        Matrix temp(rows, cols, batch);
        
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = mat[i][j] + M[i][j];

        return temp;
    }
    
    Matrix Matrix::operator- (Matrix M) const {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        Matrix temp(rows, cols, batch);

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

        Matrix temp(rows, M.get_cols(), batch);

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
        Matrix temp(rows, M.get_cols(), batch);
        
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < M.get_cols(); j++){
                vector<double> sum(batch);
                fill(sum.begin(), sum.end(), 0);
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

        Matrix temp(rows, cols, batch);
        
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = mat[i][j] * M[i][j];

        return temp;
    }

    Matrix Matrix::operator! () const {
        Matrix temp(cols, rows, batch);

        for(int i = 0; i < cols; i++)
            for(int j = 0; j < rows; j++)
                temp[i][j] = mat[j][i];
    
        return temp;
    }

    vector<double> *Matrix::operator[] (int m){
        if(m < 0 || m >= rows)
            throw -1;

        return mat[m];
    }

    Matrix::operator std::string() const {
        std::string val = "{";
        for(int i = 0; i < batch; i++){
            val += std::to_string(i) + ":{";
            for(int j = 0; j < rows; j++){
                val += "{";
                for(int k = 0; k < cols; k++){
                    val += std::to_string(mat[j][k].at(i));
                    if(k != cols - 1)
                        val += ",";
                }
                val += "}";
                if(j != rows-1)
                    val += "\n";
            }
            val += "}";
            if(i != batch - 1)
                val += "\n";
        }
        val += "}";
        return val;
    }

    FHEMatrix::FHEMatrix(Matrix mat, CryptoContext<DCRTPoly> cc, Key_t key) {
        this->rows = mat.get_rows();
        this->cols = mat.get_cols();
        this->batch = mat.get_batch();
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

    FHEMatrix FHEMatrix::operator^ (FHEMatrix M) {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        FHEMatrix temp(rows, cols, batch, cc);

        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = cc->EvalMult(mat[i][j], M[i][j]);

        return temp;
    }

    Ciphertext_t *FHEMatrix::operator[] (int m) {
        if(m < 0 || m >= rows)
            throw -1;

        return mat[m];
    }

    Matrix FHEMatrix::decrypt(Key_t key) const {
        Matrix temp(rows, cols, batch);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                Plaintext result;
                cc->Decrypt(key.secretKey, mat[i][j], &result);
                result->SetLength(batch);
                temp[i][j] = result->GetRealPackedValue();
                //std::cout << result << ",";
            }
            //std::cout << std::endl;
        }
        return temp;
    }
}

