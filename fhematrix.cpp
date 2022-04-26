#include "fhematrix.h"

namespace mat {
    Matrix::Matrix(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;

        this->mat.resize(rows, vector<double>(cols));
    }

    int Matrix::get_rows() const {
        return rows;
    }
    
    int Matrix::get_cols() const {
        return cols;
    }

    vector<vector<double>> Matrix::get_mat() {
        return mat;
    }

    void Matrix::add(double val) {
        for(vector<double>& x : mat)
            for(double& y: x)
                y += val;
    }

    void Matrix::add(Matrix *M) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] += M->at(i, j);
    }

    void Matrix::subtract(double val) {
        for(vector<double>& x : mat)
            for(double& y: x)
                y -= val;
    }

    void Matrix::subtract(Matrix *M) {
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

    void Matrix::multiply(double val) {
        for(vector<double>& x : mat)
            for(double& y: x)
                y *= val;
    }

    void Matrix::element_multiply(Matrix *M) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] *= M->at(i,j);
    }

    Matrix *Matrix::multiply(Matrix *M) const {
        Matrix *temp = new Matrix(rows, M->get_cols());
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < M->get_cols(); j++) {
                double sum = 0;
                for(int k = 0; k < cols; k++) {
                    sum += mat[i][k] * M->at(k,j);
                }
                temp->set(i,j,sum);
            }
        }
        return temp;
    }

    vector<double> Matrix::multiply(vector<double> vec) const {
        vector<double> v(rows);
        for(int i = 0; i < rows; i++) {
            double sum = 0;
            for(int j = 0; j < cols; j++) {
                sum += mat[i][j] * vec[j];
            }
            v[i] = sum;
        }
        return v;
    }

    double Matrix::at(int row, int col) const {
        return mat[row][col];
    }

    void Matrix::set(int row, int col, double val) {
        mat[row][col] = val;
    }

    Matrix *Matrix::copy() const {
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

    FHEMatrix::FHEMatrix(Matrix *mat, CryptoContext<DCRTPoly> cc, LPKeyPair<DCRTPoly> keys) {
        this->rows = mat->get_rows();
        this->cols = mat->get_cols();
        this->cc = cc;
        this->mat.resize(this->rows);
        vector<vector<double>> vecs = mat->get_mat();
        for(int i = 0; i < this->rows; i++) {
            Plaintext ptxt = cc->MakeCKKSPackedPlaintext(vecs[i]);
            Ciphertext<DCRTPoly> ctxt = cc->Encrypt(keys.publicKey, ptxt);
            this->mat[i] = ctxt;
        }
    }

    FHEMatrix::FHEMatrix(int rows, int cols, vector<Ciphertext<DCRTPoly>> mat,
            CryptoContext<DCRTPoly> cc) {
        this->rows = rows;
        this->cols = cols;
        this->cc = cc;
        this->mat = mat;
    }

    int FHEMatrix::get_rows() const {
        return rows;
    }

    int FHEMatrix::get_cols() const {
        return cols;
    }

    vector<Ciphertext<DCRTPoly>> FHEMatrix::get_mat() const {
        return mat;
    }

    void FHEMatrix::add(double val) {
        for(int i = 0; i < rows; i++) {
            Ciphertext<DCRTPoly> ctxt = cc->EvalAdd(mat[i], val);
            mat[i] = ctxt;
        }
    }

    void FHEMatrix::add(FHEMatrix *M) {
        for(int i = 0; i < rows; i++) {
            Ciphertext<DCRTPoly> ctxt = cc->EvalAdd(mat[i], M->at(i));
            mat[i] = ctxt;
        }
    }

    void FHEMatrix::subtract(double val) {
        for(int i = 0; i < rows; i++) {
            Ciphertext<DCRTPoly> ctxt = cc->EvalSub(mat[i], val);
            mat[i] = ctxt;
        }
    }

    void FHEMatrix::subtract(FHEMatrix *M) {
        for(int i = 0; i < rows; i++) {
            Ciphertext<DCRTPoly> ctxt = cc->EvalSub(mat[i], M->at(i));
            mat[i] = ctxt;
        }
    }

    void FHEMatrix::multiply(double val) {
        for(int i = 0; i < rows; i++) {
            Ciphertext<DCRTPoly> ctxt = cc->EvalMult(mat[i], val);
            mat[i] = ctxt;
        }
    }

    void FHEMatrix::element_multiply(FHEMatrix *M) {
        for(int i = 0; i < rows; i++) {
            Ciphertext<DCRTPoly> ctxt = cc->EvalMult(mat[i], M->at(i));
            mat[i] = ctxt;
        }
    }

    Ciphertext<DCRTPoly> FHEMatrix::multiply(Ciphertext<DCRTPoly> vec) const {
        vector<Ciphertext<DCRTPoly>> ct;
        
        for(int i = 0; i < rows; i++) {
            Ciphertext<DCRTPoly> result = cc->EvalInnerProduct(vec, mat[i], cols);
            ct.push_back(result);
        }

        Ciphertext<DCRTPoly> merged = cc->EvalMerge(ct);
        return merged;
    }

    Ciphertext<DCRTPoly> FHEMatrix::at(int row) const {
        return mat[row];
    }

    void FHEMatrix::set(int row, Ciphertext<DCRTPoly> vec) {
        mat[row] = vec;
    }

    Matrix *FHEMatrix::decrypt(LPKeyPair<DCRTPoly> keys) const {
        Matrix *M = new Matrix(rows, cols);
        return M;
    }
}
