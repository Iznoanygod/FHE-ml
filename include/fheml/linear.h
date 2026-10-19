// Generic encrypted linear maps for CKKS.
//
// Every linear layer of a CNN (convolution with any stride/padding, dense,
// average pooling, folded batch-norm, ...) is a matrix acting on the flattened
// activation vector that lives in the ciphertext slots. A LinearTransform
// compiles such a (sparse) matrix into its non-zero generalized diagonals and
// evaluates it with the baby-step/giant-step (BSGS) diagonal method of
// Halevi-Shoup, using hoisted rotations for the baby steps.
//
//   y[r] = sum_d diag_d[r] * x[(r + d) mod S],   diag_d[r] = M[r][(r + d) mod S]
//
// With d = g + b (b = baby step, g = giant step):
//
//   y = sum_g rot_g( sum_b rot_{-g}(diag_{g+b}) (*) rot_b(x) )
//
// so only |{b}| + |{g}| rotations are needed instead of |{d}|.
//
// For layers with few outputs (dense heads) the matrix can be lowered with the
// hybrid method of Gazelle: output r is accumulated in the slots r + j*R of a
// "folded" S x S matrix whose diagonals all fall in (-R, R), and a final
// rotate-and-sum over log2(S/R) steps adds the partial sums into slots [0, R).
#ifndef FHEML_LINEAR_H_
#define FHEML_LINEAR_H_

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "fheml/context.h"

namespace fheml {

// Row-major sparse matrix (rows x cols).
class SparseMatrix {
 public:
  SparseMatrix() = default;
  SparseMatrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), entries_(rows) {}

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t nnz() const;

  // Accumulates (duplicates are summed).
  void add(size_t r, size_t c, double v);
  const std::vector<std::pair<size_t, double>>& row(size_t r) const { return entries_[r]; }

  std::vector<double> multiply(const std::vector<double>& x) const;
  // Returns this * other (this: m x k, other: k x n).
  SparseMatrix multiply(const SparseMatrix& other) const;
  static SparseMatrix identity(size_t n);

 private:
  size_t rows_ = 0;
  size_t cols_ = 0;
  std::vector<std::vector<std::pair<size_t, double>>> entries_;
};

class LinearTransform {
 public:
  LinearTransform() = default;
  // Compiles M for ciphertexts with `slots` slots. Requires rows, cols <= slots.
  // baby_step == 0 selects the step that minimises the rotation count.
  // fold_block > 0: rows of `m` are partial sums to be added with a stride of
  // fold_block (see fold_rows()); the result is valid in slots [0, fold_block).
  LinearTransform(const SparseMatrix& m, uint32_t slots, uint32_t baby_step = 0, uint32_t fold_block = 0);

  // Re-arranges an (rows x slots) matrix whose rows are compact outputs into
  // the folded S x S form used with fold_block = R (R >= rows, power of two).
  static SparseMatrix fold_rows(const SparseMatrix& m, uint32_t slots, uint32_t block);
  // Number of diagonals of m (as an S x S slot matrix).
  static size_t count_diagonals(const SparseMatrix& m, uint32_t slots);

  Ctxt eval(const FHEContext& ctx, const Ctxt& x) const;
  // Runs exactly the same BSGS schedule on cleartext slot vectors. Used to
  // validate a compiled plan without any cryptography.
  std::vector<double> eval_slots(const std::vector<double>& x) const;

  std::vector<int32_t> rotation_indices() const;
  size_t num_diagonals() const { return num_diagonals_; }
  size_t num_baby_steps() const { return babies_.size(); }
  size_t num_giant_steps() const { return giants_.size(); }
  uint32_t baby_step() const { return baby_step_; }
  uint32_t slots() const { return slots_; }
  uint32_t fold_block() const { return fold_; }

  // Cache encoded plaintext diagonals per level (fast, but memory heavy for
  // big layers: one plaintext per diagonal).
  void set_cache_plaintexts(bool on) { cache_plaintexts_ = on; }

 private:
  struct Term {
    int32_t baby;               // index into babies_
    std::vector<double> diag;   // diag_{g+b} pre-rotated by -g
  };
  struct Giant {
    int32_t step;               // g
    std::vector<Term> terms;
  };

  uint32_t slots_ = 0;
  uint32_t baby_step_ = 1;
  size_t num_diagonals_ = 0;
  std::vector<int32_t> babies_;   // distinct baby rotations (0 allowed)
  std::vector<Giant> giants_;
  uint32_t fold_ = 0;
  bool cache_plaintexts_ = false;

  struct Cache {
    std::mutex mu;
    std::map<uint32_t, std::vector<std::vector<Ptxt>>> by_level;
  };
  std::shared_ptr<Cache> cache_ = std::make_shared<Cache>();
};

}  // namespace fheml

#endif  // FHEML_LINEAR_H_
