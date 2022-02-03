#include "fheml.h"

int main(int argc, char** argv) {
    uint32_t multDepth = 1;
    uint32_t scaleFactorBits = 50;
    uint32_t batchSize = 2;

    SecurityLevel securityLevel = HEStd_128_classic;

	CryptoContext<DCRTPoly> cc =
		CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
			multDepth, scaleFactorBits, batchSize, securityLevel);

	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);

    auto keys = cc->KeyGen();
	
    cc->EvalMultKeyGen(keys.secretKey);

    fhe::Matrix M(2,2,1);
    M[0][0] = {1};
    M[0][1] = {2};
    M[1][0] = {3};
    M[1][1] = {4};

    std::cout << std::string(M) << std::endl;

    fhe::Matrix N(2,2,2);
    N[0][0] = {1,2};
    N[0][1] = {2,4};
    N[1][0] = {3,6};
    N[1][1] = {4,8};

    std::cout << std::string(N) << std::endl;

    fhe::Matrix MpN = M+N;
    fhe::Matrix MxN = M*N;

    std::cout << std::string(MpN) << std::endl;
    std::cout << std::string(MxN) << std::endl;

    std::cout << "FHE" << std::endl;
    fhe::FHEMatrix M_e(M,2,cc,keys);
    fhe::FHEMatrix N_e(N,2,cc,keys);

    fhe::FHEMatrix MpN_e = M_e+N_e;
    fhe::FHEMatrix MxN_e = M_e*N_e;

    fhe::Matrix M_d = M_e.decrypt(keys);
    fhe::Matrix N_d = N_e.decrypt(keys);

    fhe::Matrix MpN_d = MpN_e.decrypt(keys);
    fhe::Matrix MxN_d = MxN_e.decrypt(keys);
    
    std::cout << std::string(M_d) << std::endl;
    std::cout << std::string(N_d) << std::endl;
    std::cout << std::string(MpN_d) << std::endl;
    std::cout << std::string(MxN_d) << std::endl;

    return 0;

}
