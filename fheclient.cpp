#include "fheml.h"
#include <iostream>
#include <fstream>
#include "bmpparse.h"
#include "socketio.h"
#include <unistd.h>
#include <cmath>

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
    (void) !fread(inputbuff, 1, file_length, inputctxt);
    fclose(inputctxt);
    socket_send(sockfd, inputbuff, file_length);
    free(inputbuff);
//receive file from server
    char *inter_buffer;
    int inter_length = socket_read(sockfd, &inter_buffer);
    FILE *inter_file = fopen("inter.ctxt", "w+");
    fwrite(inter_buffer, 1, inter_length, inter_file);
    fclose(inter_file);
    Serial::DeserializeFromFile("inter.ctxt", ctxt, SerType::BINARY);

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
    (void) !fread(interbuff, 1, file_length, interctxt);
    fclose(interctxt);
    socket_send(sockfd, interbuff, file_length);
    free(interbuff);
    //receive file from server
    char *final_buffer;
    int final_length = socket_read(sockfd, &final_buffer);
    FILE *final_file = fopen("final.ctxt", "w+");
    fwrite(final_buffer, 1, final_length, final_file);
    fclose(final_file);
    Serial::DeserializeFromFile("final.ctxt", ctxt, SerType::BINARY);
    delete final_buffer;
    delete inter_buffer;
    Plaintext result;
    cc->Decrypt(key.secretKey, ctxt, &result);
    result->SetLength(10);
    vector<double> result_vector = result->GetRealPackedValue();
    int max = 0;
    for(int i = 0; i < 10; i++)
        if(result_vector[i] > result_vector[max])
            max = i;
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
    char *disconnect_message = (char*) "disconnect";
    ml::Network *net = new ml::Network(784, 200, 10, 0.005);
    net->load("net.nf");
    while(1) {
        char line[4096];
        printf(">");
        (void) !scanf("%[^\n]%*c", line);
        char command[4096];
        char option[4096];
        sscanf(line, "%s %s", command, option);
        if(!strcmp(command, "disconnect")) {
            socket_send(sockfd, disconnect_message, strlen(disconnect_message));
            close(sockfd);
            break;
        }
        else if(!strcmp(command, "predict")) {
            int prediction = predict(option, sockfd, cc, keys);
            printf("prediction: %d\n", prediction);
            double *pixels = parse_bmp(option);
            vector<double> pixel_vector (pixels, pixels + 784);
            vector<double> reg_pred = net->predict(pixel_vector);
            int rmax = 0;
            for(int i = 1; i < 10; i++)
                if(reg_pred[i] > reg_pred[rmax])
                    rmax = i;
            printf("unencrypted net result: %d\n", rmax);
        }
    }

}
