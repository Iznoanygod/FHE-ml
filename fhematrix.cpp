#include "fhematrix.h"
#include <iostream>
namespace fhe {
    Matrix::Matrix(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;

        this->mat = new double*[rows];
        for(int i = 0; i < rows; i++)
            this->mat[i] = new double[cols];
    }

    int Matrix::get_rows() {
        return rows;
    }

    int Matrix::get_cols() {
        return cols;
    }

    Matrix Matrix::operator+ (Matrix M) {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }
        
        Matrix temp(rows, cols);
        
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = mat[i][j] + M[i][j];

        return temp;
    }
    
    Matrix Matrix::operator- (Matrix M) {
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
    Matrix Matrix::operator* (Matrix M) {
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
    Matrix Matrix::operator* (Matrix M) {
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
#endif

        return temp;
    }

    Matrix Matrix::operator^ (Matrix M) {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        Matrix temp(rows, cols);
        
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                temp[i][j] = mat[i][j] * M[i][j];

        return temp;
    }

    Matrix Matrix::operator! () {
        Matrix temp(cols, rows);

        for(int i = 0; i < cols; i++)
            for(int j = 0; j < rows; j++)
                temp[i][j] = mat[j][i];
    
        return temp;
    }

    double* Matrix::operator[] (int m){
        if(m < 0 || m >= rows)
            throw -1;

        return mat[m];
    }
    
}

