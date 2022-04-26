#include "bmpparse.h"
#include <stdlib.h>
#include <stdio.h>

double *parse_bmp(char *file_name) {
    FILE *input_file;
    input_file = fopen(file_name, "r");

    fseek(input_file, 0xA, SEEK_SET);
    int offset;
    fread(&offset, 4, 1, input_file);
    
    //each row is 4 bytes, but only uses 28 bits(28 first bits, last 4 bits are 0 padding)
    fseek(input_file, offset, SEEK_SET);

    double *vector = malloc(sizeof(double) * 28*28);

    double mat[28][28];
    for(int i = 0; i < 28; i++) {
        int row;
        fread(&row, 4, 1, input_file);
        row = row >> 4;
        for(int j = 0; j < 28; j++) {
            if(row & 0x1 == 1)
                mat[27-j][27-i] = 0.0;
            else
                mat[27-j][27-i] = 1.0;
        }
    }
    for(int i = 0; i < 28*28; i++) {
        vector[i] = mat[i%28][i/28];
    }
    return vector;
}
