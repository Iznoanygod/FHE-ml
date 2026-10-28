#include "fheml/backend.h"

#ifdef FHEML_WITH_FIDESLIB
#include <cuda_runtime_api.h>
#endif

namespace fheml {

#ifdef FHEML_WITH_FIDESLIB

bool gpu_backend_built() { return true; }

int gpu_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky "no device" error
    return 0;
  }
  return n;
}

#else

bool gpu_backend_built() { return false; }
int gpu_device_count() { return 0; }

#endif

namespace backend {

std::vector<Ctxt> rotate_hoisted(const CC& cc, const Ctxt& x, const std::vector<int32_t>& indices, bool parallel) {
  std::vector<Ctxt> out(indices.size());
  std::vector<int32_t> nonzero;
  for (int32_t i : indices)
    if (i != 0) nonzero.push_back(i);
  if (nonzero.empty()) {
    for (auto& c : out) c = x;
    return out;
  }
  auto digits = cc->EvalFastRotationPrecompute(x);
  const uint32_t m = cc->GetCyclotomicOrder();
#ifdef FHEML_WITH_FIDESLIB
  if (!parallel) {
    // GPU: one call hoists all rotations (the single-index form does not).
    std::vector<Ctxt> rot = cc->EvalFastRotation(x, nonzero, m, digits);
    for (size_t i = 0, j = 0; i < indices.size(); ++i) out[i] = indices[i] ? rot[j++] : x;
    return out;
  }
#endif
#pragma omp parallel for schedule(dynamic) if (parallel)
  for (size_t i = 0; i < indices.size(); ++i)
    out[i] = indices[i] ? cc->EvalFastRotation(x, indices[i], m, digits) : x;
  return out;
}

#ifdef FHEML_WITH_FIDESLIB

// FIDESlib has no EvalMultNoRelin / Relinearize: relinearize every product.
Ctxt mult_lazy(const CC& cc, const Ctxt& a, const Ctxt& b) { return cc->EvalMult(a, b); }
void relinearize(const CC&, Ctxt&) {}

// FIDESlib has no EvalPoly: rewrite in the Chebyshev basis on [-1, 1], where
// EvalChebyshevSeries applies no affine map and has the same depth.
Ctxt eval_poly(const CC& cc, const Ctxt& x, const std::vector<double>& coeffs) {
  std::vector<double> cheb = power_to_chebyshev(coeffs);
  return cc->EvalChebyshevSeries(x, cheb, -1.0, 1.0);
}

#else

Ctxt mult_lazy(const CC& cc, const Ctxt& a, const Ctxt& b) { return cc->EvalMultNoRelin(a, b); }
void relinearize(const CC& cc, Ctxt& c) { cc->RelinearizeInPlace(c); }
Ctxt eval_poly(const CC& cc, const Ctxt& x, const std::vector<double>& coeffs) { return cc->EvalPoly(x, coeffs); }

#endif

std::vector<double> power_to_chebyshev(const std::vector<double>& coeffs) {
  const size_t n = coeffs.size();
  std::vector<double> out(n, 0.0);
  // xk = x^k in the Chebyshev basis (plain convention, T_0 weight in xk[0]);
  // x T_0 = T_1 and x T_j = (T_{j+1} + T_{j-1}) / 2.
  std::vector<double> xk(n + 1, 0.0), next(n + 1);
  xk[0] = 1.0;
  for (size_t k = 0; k < n; ++k) {
    for (size_t j = 0; j <= k; ++j) out[j] += coeffs[k] * xk[j];
    std::fill(next.begin(), next.end(), 0.0);
    for (size_t j = 0; j <= k; ++j) {
      if (j == 0) {
        next[1] += xk[0];
      } else {
        next[j + 1] += xk[j] / 2;
        next[j - 1] += xk[j] / 2;
      }
    }
    xk.swap(next);
  }
  if (n) out[0] *= 2.0;  // OpenFHE evaluates c0 / 2
  return out;
}

}  // namespace backend
}  // namespace fheml
