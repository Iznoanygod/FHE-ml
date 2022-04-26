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
    int port = atoi(argv[1]);
    //char *directory = argv[2];
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
        printf("Waiting for client...\n");
        int sock = accept(serverFD, (struct sockaddr*) &address, (socklen_t*)&addrlen);
        if(sock < 0){
            printf("Error accepting client\n");
            return 0;
        }
        printf("Accepted connection\n");
        while(1) {
            
        }
    }
}
