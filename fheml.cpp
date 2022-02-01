#include <iostream>

#define PROFILE

#include "palisade.h"
#include "fhematrix.h"

int main(int argc, char** argv){
	
	uint32_t multDepth = 1;
	uint32_t scaleFactorBits = 50;
	uint32_t batchSize = 8;
	
	SecurityLevel securityLevel = HEStd_128_classic;

	CryptoContext<DCRTPoly> cc =
		CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
			multDepth, scaleFactorBits, batchSize, securityLevel);

	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);

    auto keys = cc->KeyGen();
	
    cc->EvalMultKeyGen(keys.secretKey);

	vector<double> x1 = {0.25, 0.5, 0.75, 0.0, 2.0, 3.0, 4.0, 5.0};
	vector<double> x2 = {5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25};

	Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	
	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	Ciphertext_t c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	
    auto cAdd = cc->EvalAdd(c1, c2);
	auto cMul = cc->EvalMult(c1, c2);

	Plaintext result;

	cc->Decrypt(keys.secretKey, cAdd, &result);
	result->SetLength(batchSize);
	std::cout << "x1 + x2 = " << result;

	cc->Decrypt(keys.secretKey, cMul, &result);
	result->SetLength(batchSize);
	std::cout << "x1 * x2 = " << result;
    
    fhe::Matrix M(2,2);
    M[0][0] = 1;
    M[0][1] = 1;
    M[1][0] = 1;
    M[1][1] = 1;
    fhe::Matrix N(2,2);
    N[0][0] = 1;
    N[0][1] = 0;
    N[1][0] = 0;
    N[1][1] = 1;
    fhe::Matrix O = M*N;
    std::cout << O[0][0] << std::endl;
    fhe::Matrix L = !O;
    std::cout << L[1][0] << std::endl;
    return 0;
}
