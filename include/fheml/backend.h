// CKKS backend selection.
//
// fheml is written against OpenFHE's CryptoContext API. Built with
// -DFHEML_GPU=ON it uses FIDESlib instead (https://github.com/CAPS-UMU/FIDESlib),
// a CUDA CKKS library whose `fideslib::` API mirrors OpenFHE's and wraps a
// (patched) OpenFHE 1.6.1 context: it produces bit-identical ciphertexts and
// falls back to that OpenFHE context when no GPU is used. The few operations
// FIDESlib does not expose are emulated by the helpers below.
#ifndef FHEML_BACKEND_H_
#define FHEML_BACKEND_H_

#include <cstdint>
#include <vector>

#include "openfhe.h"

#ifdef FHEML_WITH_FIDESLIB
#include "fideslib.hpp"
#endif

namespace fheml {

#ifdef FHEML_WITH_FIDESLIB
namespace fhe = fideslib;
#else
namespace fhe = lbcrypto;
#endif

using CC = fhe::CryptoContext<fhe::DCRTPoly>;
using Ctxt = fhe::Ciphertext<fhe::DCRTPoly>;
using Ptxt = fhe::Plaintext;

// True when this build links FIDESlib.
bool gpu_backend_built();
// Number of visible CUDA devices (0 without FIDESlib or without a GPU).
int gpu_device_count();

namespace backend {

// Rotations of x by each index, sharing one digit decomposition (hoisting).
// Index 0 returns x itself. On the GPU this is one batched call; on the CPU
// the rotations run in parallel when `parallel` is set.
std::vector<Ctxt> rotate_hoisted(const CC& cc, const Ctxt& x, const std::vector<int32_t>& indices, bool parallel);

// a * b, left unrelinearized when the backend allows it (OpenFHE). Sum such
// products and call relinearize() once.
Ctxt mult_lazy(const CC& cc, const Ctxt& a, const Ctxt& b);
void relinearize(const CC& cc, Ctxt& c);

// sum_k coeffs[k] x^k (Paterson-Stockmeyer, same depth as OpenFHE EvalPoly).
Ctxt eval_poly(const CC& cc, const Ctxt& x, const std::vector<double>& coeffs);

// Power-basis coefficients -> Chebyshev coefficients in OpenFHE's convention
// (p(x) = c0/2 + sum_{k>=1} c_k T_k(x)), valid for every x.
std::vector<double> power_to_chebyshev(const std::vector<double>& coeffs);

}  // namespace backend
}  // namespace fheml

#endif  // FHEML_BACKEND_H_
