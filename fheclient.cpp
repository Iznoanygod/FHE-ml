#include "fheml.h"
#include <iostream>
#include <fstream>
#include "bmpparse.h"
#include "socketio.h"

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


    Plaintext in_between;
    cc->Decrypt(key.secretKey, ctxt, &in_between);
    in_between->SetLength(200);
    vector<double> in_b = in_between->GetRealPackedValue();
    
    for(int i = 0; i < 200; i++)
        in_b[i] = 1 / (1 + exp(-in_b[i]));

    in_between = cc->MakeCKKSPackedPlaintext(in_b);
    ctxt = cc->Encrypt(key.publicKey, in_between);
    
    //send to server, perform round 2
    Serial::SerializeToFile("inter.ctxt", ctxt, SerType::BINARY);

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
    /*usint m = 8192;
    usint init_size = 3;
    usint dcrtBits = 40;
    auto cc =
        CryptoContextFactory<DCRTPoly>::genCryptoContextCKKSWithParamsGen(
            m, init_size,
            dcrtBits, 10,
            784,
            OPTIMIZED, 20, 10,
            FIRSTMODSIZE, BV, APPROXAUTO);
        
    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);
     
    auto key = cc->KeyGen();

    cc->EvalMultKeyGen(key.secretKey);
    cc->EvalSumKeyGen(key.secretKey);

    vector<int> rotations(200);
    for(int i = 1; i < 201; i++)
        rotations[i-1] = -i;
    cc->EvalAtIndexKeyGen(key.secretKey, rotations);
    */
    /*Serial::SerializeToFile("cryptocontext.cf", cc, SerType::BINARY);
    Serial::SerializeToFile("public.kf", key.publicKey, SerType::BINARY);
    Serial::SerializeToFile("private.kf", key.secretKey, SerType::BINARY);
    std::ofstream multKeyFile("multkey.kf", std::ios::out | std::ios::binary);
    cc->SerializeEvalMultKey(multKeyFile, SerType::BINARY);
    std::ofstream rotationKeyFile("rotkey.kf", std::ios::out | std::ios::binary);
    cc->SerializeEvalAutomorphismKey(rotationKeyFile, SerType::BINARY);
    std::ofstream sumKeyFile("sumkey.kf", std::ios::out | std::ios::binary);
    cc->SerializeEvalSumKey(sumKeyFile, SerType::BINARY);
    */
    CryptoContext<DCRTPoly> cc;
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
    Serial::DeserializeFromFile("cryptocontext.cf", cc, SerType::BINARY);
    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);
    LPKeyPair<DCRTPoly> keys;
    Serial::DeserializeFromFile("public.kf", keys.publicKey, SerType::BINARY);
    Serial::DeserializeFromFile("private.kf", keys.secretKey, SerType::BINARY);
    ml::Network *net = new ml::Network(784, 200, 10, 0.005);
    net->load("net.nf");

    ml::FHENetwork *fhenet = new ml::FHENetwork(net, cc, keys);
    fhenet->save("netweights");
    while(1) {
        char line[4096];
        scanf("%[^\n]%*c", line);
        break;    
    }

}
