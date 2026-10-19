# Research notes: homomorphic convolution and CNN inference

This survey informed the design of fheml (see [DESIGN.md](DESIGN.md)).

**How the citations were checked.** Citations in sections 2 and 3 were checked
against the papers' pages in October 2026. The foundational citations in
section 1 are well known and were not re-fetched. Items marked *(unverified)*
were not confirmed.

## 1. Foundations

- **CryptoNets.** Gilad-Bachrach, Dowlin, Laine, Lauter, Naehrig, Wernsing. ICML 2016.
  - Batch packing: each slot holds the same pixel of a different image.
  - Activations are x², and pooling is scaled mean pooling.
  - It gives high throughput but poor latency. Our MNIST example uses its
    shape: one ciphertext per image, and a quadratic activation.
- **Gazelle.** Juvekar, Vaikuntanathan, Chandrakasan. USENIX Security 2018, arXiv:1801.05507.
  - Packs an image into slots and computes a 2-D convolution as k² rotations
    times plaintext kernel masks.
  - Its *hybrid* matrix-vector method folds rows into partial sums and then
    applies rotate-and-sum.
  - fheml uses that hybrid method for dense layers with few outputs. Gazelle
    itself is a hybrid HE + garbled-circuits protocol.
- **Halevi–Shoup diagonal method.** *Algorithms in HElib*, CRYPTO 2014, and
  *Faster Homomorphic Linear Transformations in HElib*, CRYPTO 2018, ePrint 2018/244.
  - The matrix-vector product is Σ diag_i ⊙ rot_i(v).
  - With baby-step/giant-step it needs about 2√n rotations. Hoisting shares
    one digit decomposition across all baby-step rotations.
  - This is the core of `LinearTransform`.
- **LoLa.** Brutzkus, Elisha, Gilad-Bachrach. ICML 2019, arXiv:1812.10659.
  - Latency-oriented: it changes the packing between layers (dense, sparse,
    stacked, interleaved).
- **CHET.** Dathathri et al. PLDI 2019.
  - A compiler that selects layouts (HW-tiled vs. CHW) and CKKS parameters.
- **nGraph-HE / nGraph-HE2.** Boemer et al. CF 2019, arXiv:1810.10121; WAHC 2019, arXiv:1908.04172.
  - Graph compilers on SEAL CKKS, with lazy rescaling and plaintext-value
    optimisations.
- **Kim et al.** *Secure human action recognition by encrypted neural network
  inference*. Nature Communications 13, 2022, arXiv:2104.09164.
  - CKKS inference with a specialised packing for 1-D/temporal convolution.

## 2. Deep CNNs on CKKS with bootstrapping

- **Lee et al., ResNet-20 on RNS-CKKS.** IEEE Access 2022, arXiv:2106.07229 / ePrint 2021/783.
  - The first full ResNet-20 under standard CKKS parameters, with
    bootstrapping and a high-degree ReLU approximation.
- **Multiplexed parallel convolutions.** E. Lee et al. ICML 2022, ePrint 2021/1688.
  - Interleaves channels so that strided outputs stay dense in the slots, and
    runs several convolutions in parallel in one ciphertext.
  - About 4.7× faster ResNet-20. This is the natural next step after fheml's
    gapped layout.
- **Minimax composite ReLU.**
  - E. Lee, J.-W. Lee, J.-S. No, Y.-S. Kim. IEEE TDSC 2022, ePrint 2020/834.
  - J. Lee et al. IEEE Access 2023, arXiv:2105.10879.
  - ReLU(x) = x(1 + sign(x))/2, with sign approximated by composing low-degree
    minimax polynomials. This is much shallower than one high-degree
    polynomial.
- **HyPHEN.** D. Kim et al. IEEE Access 2024, arXiv:2302.02407.
  - Two conv methods: replication-aligned and channel-aligned. Two packings:
    gap packing and PRCR.
  - Together they cut rotations and bootstraps.
- **NeuJeans.** Ju et al. ACM CCS 2024, arXiv:2312.04356.
  - Encodes data as "coefficients-in-slot", fusing convolution into the
    CoeffToSlot step of bootstrapping.
  - This needs access to OpenFHE's bootstrapping internals.
- **Orion.** Ebel, Garimella, Reagen. ASPLOS 2025 (best paper), arXiv:2311.03470.
  - A PyTorch→FHE compiler. It computes any convolution as one BSGS linear
    transform over a multiplexed packing.
  - It places bootstraps automatically by solving a shortest path over levels.
  - fheml's "any linear layer is one BSGS matrix" design follows Orion.
- **Fhelipe.** Krastev et al. PLDI 2024.
  - A tensor compiler with automatic packing and bootstrap placement.
- **HECATE / ELASM / DaCapo / HALO.** Lee et al. CGO 2022, USENIX Security 2023 and 2024, ASPLOS 2025.
  - Scale management and bootstrap placement.
  - OpenFHE's `FLEXIBLEAUTO` covers most of the scale management.
- **AutoFHE.** Ao, Boddeti. USENIX Security 2024, arXiv:2310.08012.
  - Searches per-layer activation degrees and bootstrap placement.
- **SmartPAF.** Tong et al. MLSys 2024, arXiv:2404.03216.
  - Trains or fine-tunes with the polynomial activation in place, so that
    degree 7–27 replaces degree ~59–200.
- **HElayers / tile tensors.** Aharoni et al. PETS 2023, arXiv:2011.01805.
  - A generic tiled tensor abstraction whose tile shapes are chosen by an
    optimiser.
- **Cheetah.** Huang et al. USENIX Security 2022, ePrint 2022/207.
  - Coefficient encoding turns convolution into a single polynomial
    multiplication. It is 2PC (HE + OT), not pure FHE.

## 3. OpenFHE specifics

- **Rotations.**
  - Hoisted rotations: `EvalFastRotationPrecompute` + `EvalFastRotation`. fheml
    uses them for the baby steps.
  - `EvalRotateKeyGen(sk, indices)` should get only the indices the network
    needs, because rotation keys dominate memory.
- **Polynomials.**
  - `EvalPoly` and `EvalChebyshevSeries` / `EvalChebyshevFunction` use
    Paterson–Stockmeyer.
  - Measured depths with OpenFHE v1.4.2 on [-1, 1]:

    | Degree | ≤2 | ≤5 | ≤13 | ≤27 | ≤59 | ≤119 | ≤247 |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Levels | 2 | 3 | 4 | 5 | 6 | 7 | 8 |

  - Any other interval costs one more level, for the affine map onto [-1, 1].
- **Bootstrapping.**
  - Setup is `EvalBootstrapSetup(levelBudget, dim1, slots)` then
    `EvalBootstrapKeyGen`.
  - Sparse-slot bootstrapping, with slots < N/2, is cheaper.
  - `EvalBootstrap` returns its input unchanged if that ciphertext has more
    levels left than bootstrapping would give. So fheml encrypts the input at
    a deep level (`Network::input_level`), which makes the first bootstrap
    actually refresh.
  - Slot values must be roughly in [-1, 1].
- **Functional bootstrapping.** Alexandru, Kim, Polyakov. CRYPTO 2025, ePrint 2024/1623.
  - Targets lookup tables on integer-like inputs, so it is of limited use for
    real-valued ReLU.
- **openfhe-numpy.** A NumPy-like Python layer over OpenFHE (2025). Whether it
  supports convolution is *(unverified)*.
- **Closest OpenFHE references.**
  - Rovida & Leporati. *Encrypted Image Classification with Low Memory
    Footprint using FHE*, IJNS 2024, ePrint 2024/460,
    github.com/narger-ef/LowMemoryFHEResNet20.
    - ResNet-20 on CIFAR-10: 91.7% in under 5 minutes.
    - Gazelle-style convolution, Chebyshev ReLU of degree 59–200, and slots
      halved at each downsampling.
  - FHEON. Njungle, Jahns, Kinsy, arXiv:2510.03996, 2025 (venue unconfirmed).
    - A configurable OpenFHE CNN framework.
