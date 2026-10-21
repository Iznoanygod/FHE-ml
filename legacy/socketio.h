#ifndef _SOCKET_IO_H_
#define _SOCKET_IO_H_

int socket_connect(char* ip, int port);

int socket_send(int sockfd, char* message, int len);

int socket_read(int sockfd, char** buffer);

#endif
