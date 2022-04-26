#include "socketio.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>

int socket_connect(char* ip, int port) {
    struct sockaddr_in serv_addr;
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &serv_addr.sin_addr);
    connect(sock, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
    return sock;
}

int socket_send(int sockfd, char* message, int len) {
    int writeout = 0;
    while(1) {
        int status = send(sockfd,((char*) &len) + writeout, sizeof(len) - writeout, 0);
        if(status == -1)
            return -1;
        writeout += status;
        if(writeout == sizeof(len) || status == 0)
            break;
    }

    writeout = 0;
    while(1) {
        int status = send(sockfd, message + writeout, len - writeout, 0);
        if(status = -1)
            return -1;
        writeout += status;
        if(writeout == len || status == 0)
            break;
    }
    return writeout;
}

int socket_read(int sockfd, char** buffer) {
    int len;
    int readin=0;
    while(1){
        int status = recv(sockfd, ((char*)&len) + readin, sizeof(len)-readin, 0);
        if(status == -1){
            return 0;
        }
        readin += status;
        if(readin == sizeof(len) || status == 0)
            break;
    }
    
    readin = 0;
    char *readbuffer = new char[len+1];

    while(1){
        int status = recv(sockfd, readbuffer+readin, len-readin, 0);
        if(status == -1){
            return readin;
        }
        readin += status;
        if(readin == len || status == 0)
            break;
    }
    *buffer = readbuffer;
    return readin;
}
