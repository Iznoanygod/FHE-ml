#include "fheml.h"
#include <iostream>
#include <fstream>
#include "bmpparse.h"
#include "socketio.h"
#include <unistd.h>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "scheme/ckks/ckks-ser.h"
#include "pubkeylp-ser.h"


int predict(char *image_name, int sockfd,
        CryptoContext<DCRTPoly> cc, LPKeyPair<DCRTPoly> key) {
    double *pixels = parse_bmp(image_name);
    vector<double> pixel_vector (pixels, pixels + 784);
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(pixel_vector);
    Ciphertext<DCRTPoly> ctxt = cc->Encrypt(key.publicKey, ptxt);
    //send to server, perform round 1
    
    char *message = (char*) "inference_1";
    socket_send(sockfd, message, strlen(message));
    Serial::SerializeToFile("input.ctxt", ctxt, SerType::BINARY);
    int file_length;
    FILE *inputctxt = fopen("input.ctxt", "r");
    fseek(inputctxt, 0, SEEK_END);
    file_length = ftell(inputctxt);
    rewind(inputctxt);
    char* inputbuff = (char*) malloc(file_length);
    fread(inputbuff, 1, file_length, inputctxt);
    fclose(inputctxt);
    socket_send(sockfd, inputbuff, file_length);
    free(inputbuff);
//receive file from server
    

    Plaintext in_between;
    cc->Decrypt(key.secretKey, ctxt, &in_between);
    in_between->SetLength(200);
    vector<double> in_b = in_between->GetRealPackedValue();
    
    for(int i = 0; i < 200; i++)
        in_b[i] = 1 / (1 + exp(-in_b[i]));

    in_between = cc->MakeCKKSPackedPlaintext(in_b);
    ctxt = cc->Encrypt(key.publicKey, in_between);
    
    //send to server, perform round 2
    message = (char*) "inference_2";
    socket_send(sockfd, message, strlen(message));
    Serial::SerializeToFile("inter.ctxt", ctxt, SerType::BINARY);
    FILE *interctxt = fopen("inter.ctxt", "r");
    fseek(inputctxt, 0, SEEK_END);
    file_length = ftell(interctxt);
    rewind(interctxt);
    char* interbuff = (char*) malloc(file_length);
    fread(interbuff, 1, file_length, interctxt);
    fclose(interctxt);
    socket_send(sockfd, interbuff, file_length);
    free(interbuff);
    //receive file from server

    Plaintext result;
    cc->Decrypt(key.secretKey, ctxt, &result);
    result->SetLength(10);
    vector<double> result_vector = result->GetRealPackedValue();
    int max = 0;
    for(int i = 0; i < 10; i++)
        if(result_vector[i] > result_vector[max])
            max = 1;
    return max;
}

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Usage: fheclient ip port\n");
        return 0;
    }
    printf("Loading contexts...");
    CryptoContext<DCRTPoly> cc;
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    cc->ClearEvalSumKeys();
    lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
    Serial::DeserializeFromFile("cryptocontext.cf", cc, SerType::BINARY);
    LPKeyPair<DCRTPoly> keys;
    Serial::DeserializeFromFile("public.kf", keys.publicKey, SerType::BINARY);
    Serial::DeserializeFromFile("private.kf", keys.secretKey, SerType::BINARY);

    //std::ifstream multKeyFile("multkey.kf", std::ios::in | std::ios::binary);
    //std::ifstream rotKeyFile("rotkey.kf", std::ios::in | std::ios::binary);
    //std::ifstream sumKeyFile("sumkey.kf", std::ios::in | std::ios::binary);
    //cc->DeserializeEvalMultKey(multKeyFile, SerType::BINARY);
    //cc->DeserializeEvalAutomorphismKey(rotKeyFile, SerType::BINARY);
    //cc->DeserializeEvalSumKey(sumKeyFile, SerType::BINARY);
    printf("Done\n");

    printf("Connecting to server...\n");
    int sockfd;
    do{
        sockfd = socket_connect(argv[1], atoi(argv[2]));
        if(sockfd < 0)
            sleep(5);
    }while(sockfd < 0);
    printf("Connected\n");
    while(1) {
        char line[4096];
        printf(">");
        scanf("%[^\n]%*c", line);
        
        break;    
    }

}
