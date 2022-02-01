#include "fhematrix.h"
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
                ;

        return temp;
    }
    
    Matrix Matrix::operator- (Matrix M) {
        if(rows != M.get_rows() || cols != M.get_cols()) {
            throw -1;
        }

        Matrix temp(rows, cols);

        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                ;

        return temp;
    }

    Matrix Matrix::operator* (Matrix M){
        if(cols != M.get_rows()){
            return -1;
        }


    }
}

