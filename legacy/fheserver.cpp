#include "fheml.h"
#include <iostream>
#include <fstream>
#include "socketio.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "scheme/ckks/ckks-ser.h"
#include "pubkeylp-ser.h"

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Missing arguments\n");
        printf("Usage: fheserver port netdir\n");
        return 0;
    }
    
    std::string net_dir = argv[2];
    printf("Loading contexts...");
    CryptoContext<DCRTPoly> cc;
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    cc->ClearEvalSumKeys();
    lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
    Serial::DeserializeFromFile("cryptocontext.cf", cc, SerType::BINARY);
    LPKeyPair<DCRTPoly> keys;
    Serial::DeserializeFromFile("public.kf", keys.publicKey, SerType::BINARY);

    std::ifstream multKeyFile("multkey.kf", std::ios::in | std::ios::binary);
    std::ifstream rotKeyFile("rotkey.kf", std::ios::in | std::ios::binary);
    std::ifstream sumKeyFile("sumkey.kf", std::ios::in | std::ios::binary);
    cc->DeserializeEvalMultKey(multKeyFile, SerType::BINARY);
    cc->DeserializeEvalAutomorphismKey(rotKeyFile, SerType::BINARY);
    cc->DeserializeEvalSumKey(sumKeyFile, SerType::BINARY);
    printf("Done\n");

    printf("Creating encrypted network...");
    ml::FHENetwork *fhenet = new ml::FHENetwork(net_dir, cc, keys, 784, 200, 10, 0.005);
    printf("Done\n");

    int port = atoi(argv[1]);
    int opt = 0;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int serverFD = socket(AF_INET, SOCK_STREAM, 0);
    if(serverFD == 0){
        printf("Server File Descriptor Error\n");
        return 1;
    }
    int sockopt = setsockopt(serverFD, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    if(sockopt > 0) {
        printf("Failed to setsockopt\n");
        return 1;
    }
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons(port);
    int bindret = bind(serverFD, (struct sockaddr*)&address, sizeof(address));
    if(bindret < 0){
        printf("Failed to bind to socket\n");
        return 1;
    }
    printf("Bound to socket...\n");
    
    if (listen(serverFD, 3) < 0) 
    { 
        printf("Failed to listen.\n"); 
        return 1;
    } 

    while(1) {
ACCEPTCLIENT:
        printf("Waiting for client...\n");
        int sock = accept(serverFD, (struct sockaddr*) &address, (socklen_t*)&addrlen);
        if(sock < 0){
            printf("Error accepting client\n");
            return 0;
        }
        printf("Accepted connection\n");
        while(1) {
            char *input_message;
            socket_read(sock, &input_message);
            if(!strcmp(input_message, "inference_1")) {
                printf("Performing first inference\n");
                char *ctxt_message;
                int length = socket_read(sock, &ctxt_message);
                FILE *ctxt_file = fopen("input.ctxt", "w+");
                fwrite(ctxt_message, 1, length, ctxt_file);
                fclose(ctxt_file);
                Ciphertext<DCRTPoly> input;
                Serial::DeserializeFromFile("input.ctxt", input, SerType::BINARY);
                printf("Received input ciphertext\n");
                Ciphertext<DCRTPoly> inter = fhenet->first_predict(input);
                printf("Inference complete\n");
                Serial::SerializeToFile("inter.ctxt", inter, SerType::BINARY);
                FILE *int_file = fopen("inter.ctxt", "r");
                fseek(int_file, 0, SEEK_END);
                length = ftell(int_file);
                rewind(int_file);
                char *inter_buffer = new char[length];
                (void) !fread(inter_buffer, 1, length, int_file);
                fclose(int_file);
                socket_send(sock, inter_buffer, length);
                printf("Result sent\n");
                delete inter_buffer;
                delete ctxt_message;
            }
            else if(!strcmp(input_message, "inference_2")) {
                printf("Performing second inference\n");
                char *ctxt_message;
                int length = socket_read(sock, &ctxt_message);
                FILE *ctxt_file = fopen("input.ctxt", "w+");
                fwrite(ctxt_message, 1, length, ctxt_file);
                fclose(ctxt_file);
                Ciphertext<DCRTPoly> input;
                Serial::DeserializeFromFile("input.ctxt", input, SerType::BINARY);
                printf("Received input ciphertext\n");
                Ciphertext<DCRTPoly> inter = fhenet->second_predict(input);
                printf("Inference complete\n");
                Serial::SerializeToFile("inter.ctxt", inter, SerType::BINARY);
                FILE *int_file = fopen("inter.ctxt", "r");
                fseek(int_file, 0, SEEK_END);
                length = ftell(int_file);
                rewind(int_file);
                char *inter_buffer = new char[length];
                (void) !fread(inter_buffer, 1, length, int_file);
                fclose(int_file);
                socket_send(sock, inter_buffer, length);
                printf("Result sent\n");
                delete inter_buffer;
                delete ctxt_message;
            }
            else if(!strcmp(input_message, "disconnect")) {
                close(sock);
                goto ACCEPTCLIENT;
            }
            delete input_message;
        }
    }
}
