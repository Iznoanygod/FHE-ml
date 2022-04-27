#include "bmpparse.h"
#include <stdlib.h>
#include <stdio.h>

double *parse_bmp(char *file_name) {
    FILE *input_file;
    input_file = fopen(file_name, "r");

    fseek(input_file, 0xA, SEEK_SET);
    int offset;
    (void)!fread(&offset, 4, 1, input_file);
    
    //each pixel is 8 bits(1 byte)
    fseek(input_file, offset, SEEK_SET);

    double *vector = new double[28*28];

    double mat[28][28];
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            unsigned char pixel;
            (void)!fread(&pixel, 1, 1, input_file);
            pixel = 0xFF - pixel;
            double pixel_value = pixel / 255.0;
            mat[27-i][j] = pixel_value;
        }
    }
    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++) {
            if(mat[i][j] < 0.1)
                printf(" ");
            else if(mat[i][j] < 0.5) {
                printf("%%");
            }
            else
                printf("#");
        }
        printf("\n");
    }
    for(int i = 0; i < 28*28; i++) {
        vector[i] = mat[i/28][i%28];
    }
    fclose(input_file);
    return vector;
}
    
