// Python bindings: `import fheml`.
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fheml/fheml.h"

namespace py = pybind11;
using namespace fheml;

namespace {

// Opaque handle so Python never touches OpenFHE types directly.
struct Ciphertext {
  Ctxt ct;
  uint32_t level() const { return ct->GetLevel(); }
};

using Vec = std::vector<double>;
using LayerPtr = std::shared_ptr<Layer>;

}  // namespace

PYBIND11_MODULE(_fheml, m) {
  m.doc() = "Encrypted neural-network inference on OpenFHE CKKS";

  py::class_<Shape>(m, "Shape")
      .def(py::init([](int c, int h, int w) { return Shape{c, h, w}; }), py::arg("c"), py::arg("h") = 1,
           py::arg("w") = 1)
      .def_readwrite("c", &Shape::c)
      .def_readwrite("h", &Shape::h)
      .def_readwrite("w", &Shape::w)
      .def("size", &Shape::size)
      .def("__repr__", [](const Shape& s) { return "Shape(" + s.str() + ")"; });

  py::class_<ContextOptions>(m, "ContextOptions")
      .def(py::init<>())
      .def_readwrite("mult_depth", &ContextOptions::mult_depth)
      .def_readwrite("slots", &ContextOptions::slots)
      .def_readwrite("scale_mod_bits", &ContextOptions::scale_mod_bits)
      .def_readwrite("first_mod_bits", &ContextOptions::first_mod_bits)
      .def_readwrite("ring_dim", &ContextOptions::ring_dim)
      .def_readwrite("secure", &ContextOptions::secure)
      .def_readwrite("bootstrapping", &ContextOptions::bootstrapping)
      .def_readwrite("level_budget", &ContextOptions::level_budget)
      .def_readwrite("rotation_window", &ContextOptions::rotation_window)
      .def_readwrite("scheme_switching", &ContextOptions::scheme_switching)
      .def_readwrite("fhew_log_q", &ContextOptions::fhew_log_q)
      .def_readwrite("bootstrap_iterations", &ContextOptions::bootstrap_iterations)
      .def_readwrite("bootstrap_precision", &ContextOptions::bootstrap_precision)
      .def_readwrite("key_switch_digits", &ContextOptions::key_switch_digits)
      .def_readwrite("use_gpu", &ContextOptions::use_gpu)
      .def_readwrite("gpu_devices", &ContextOptions::gpu_devices);

  py::class_<Ciphertext>(m, "Ciphertext").def_property_readonly("level", &Ciphertext::level);

  py::class_<FHEContext>(m, "Context")
      .def(py::init<const ContextOptions&>(), py::call_guard<py::gil_scoped_release>())
      .def("keygen", &FHEContext::keygen, py::arg("rotations"), py::call_guard<py::gil_scoped_release>())
      .def(
          "encrypt", [](const FHEContext& c, const Vec& v, uint32_t level) { return Ciphertext{c.encrypt(v, level)}; },
          py::arg("values"), py::arg("level") = 0)
      .def(
          "decrypt", [](const FHEContext& c, const Ciphertext& x, size_t n) { return c.decrypt(x.ct, n); },
          py::arg("ct"), py::arg("n"))
      .def_property_readonly("slots", &FHEContext::slots)
      .def_property_readonly("ring_dim", &FHEContext::ring_dim)
      .def_property_readonly("total_depth", &FHEContext::total_depth)
      .def_property_readonly("rotation_key_count", &FHEContext::rotation_key_count)
      .def_property_readonly("rotation_key_bytes", &FHEContext::rotation_key_bytes)
      .def("has_rotation_key", &FHEContext::has_rotation_key, py::arg("k"))
      .def("rotation_cost", &FHEContext::rotation_cost, py::arg("k"))
      .def(
          "rotation_plan",
          [](const FHEContext& c, const std::vector<int32_t>& ks) {
            auto p = c.plan_rotations(ks);
            py::dict d;
            d["key_switches"] = p->key_switches();
            d["decompositions"] = p->decompositions();
            d["max_depth"] = p->max_depth();
            return d;
          },
          py::arg("ks"), "Cost of rotate_many(c, ks): key switches, digit decompositions and depth of its rotation tree.")
      .def_property_readonly("on_gpu", &FHEContext::on_gpu)
      .def("describe", &FHEContext::describe)
      .def_static("ring_dim_for", &FHEContext::ring_dim_for, py::arg("options"));

  py::class_<Layer, LayerPtr>(m, "Layer")
      .def_property_readonly("type", &Layer::type)
      .def("describe", &Layer::describe)
      .def_property_readonly("depth", &Layer::depth)
      .def_property_readonly("in_shape", &Layer::in_shape)
      .def_property_readonly("out_shape", &Layer::out_shape)
      .def("forward_plain", &Layer::forward_plain)
      .def("params", &Layer::params,
           "Trainable values: weights then biases (linear layers), gamma then beta (layernorm), the four "
           "projections' params (attention); empty for other layers.")
      .def("set_params", &Layer::set_params, py::arg("params"))
      .def_property_readonly("trainable", &Layer::trainable);

  py::class_<LinearLayer, Layer, std::shared_ptr<LinearLayer>>(m, "LinearLayer");
  py::class_<Conv2d, LinearLayer, std::shared_ptr<Conv2d>>(m, "Conv2d")
      .def(py::init<int, int, int, int, int, int, Vec, Vec>(), py::arg("in_channels"), py::arg("out_channels"),
           py::arg("kh"), py::arg("kw"), py::arg("stride"), py::arg("pad"), py::arg("weights"),
           py::arg("bias") = Vec{});
  py::class_<Dense, LinearLayer, std::shared_ptr<Dense>>(m, "Dense")
      .def(py::init<int, int, Vec, Vec>(), py::arg("inputs"), py::arg("outputs"), py::arg("weights"),
           py::arg("bias") = Vec{});
  py::class_<AvgPool2d, LinearLayer, std::shared_ptr<AvgPool2d>>(m, "AvgPool2d")
      .def(py::init<int, int>(), py::arg("k"), py::arg("stride"));
  py::class_<MaxPool2d, Layer, std::shared_ptr<MaxPool2d>>(m, "MaxPool2d")
      .def(py::init<int, int, double>(), py::arg("k"), py::arg("stride"), py::arg("range") = 1.0,
           "Exact max pooling through CKKS<->FHEW scheme switching; inputs must lie in [-range, range].");
  py::class_<ChannelAffine, LinearLayer, std::shared_ptr<ChannelAffine>>(m, "ChannelAffine")
      .def(py::init<Vec, Vec>(), py::arg("scale"), py::arg("shift"));
  py::class_<Scale, LinearLayer, std::shared_ptr<Scale>>(m, "Scale").def(py::init<Vec>(), py::arg("factors"));
  py::class_<TokenDense, LinearLayer, std::shared_ptr<TokenDense>>(m, "TokenDense")
      .def(py::init<int, int, int, Vec, Vec>(), py::arg("tokens"), py::arg("inputs"), py::arg("outputs"),
           py::arg("weights"), py::arg("bias") = Vec{});
  py::class_<TokenMean, LinearLayer, std::shared_ptr<TokenMean>>(m, "TokenMean")
      .def(py::init<int, int>(), py::arg("tokens"), py::arg("dim"));
  py::class_<Square, Layer, std::shared_ptr<Square>>(m, "Square").def(py::init<>());
  py::class_<PolyActivation, Layer, std::shared_ptr<PolyActivation>>(m, "Poly")
      .def(py::init<Vec>(), py::arg("coeffs"));
  py::class_<ChebyshevActivation, Layer, std::shared_ptr<ChebyshevActivation>>(m, "Chebyshev")
      .def(py::init<std::string, double, double, uint32_t>(), py::arg("fn"), py::arg("a"), py::arg("b"),
           py::arg("degree"))
      .def(py::init<std::string, double, double, uint32_t, double, double, double>(), py::arg("fn"), py::arg("a"),
           py::arg("b"), py::arg("degree"), py::arg("eval_a"), py::arg("eval_b"), py::arg("out_scale") = 1.0);
  py::class_<ExactReLU, Layer, std::shared_ptr<ExactReLU>>(m, "ExactReLU")
      .def(py::init<double, double, bool>(), py::arg("range") = 1.0, py::arg("out_scale") = 1.0,
           py::arg("after_bootstrap") = false,
           "Exact ReLU: sign through CKKS<->FHEW scheme switching; inputs must lie in [-range, range]. "
           "after_bootstrap: inputs are past a CKKS bootstrap, the layer then costs one level "
           "(set by place_bootstraps).")
      .def_property_readonly("range", &ExactReLU::range)
      .def_property("after_bootstrap", &ExactReLU::after_bootstrap, &ExactReLU::set_after_bootstrap);
  py::class_<Bootstrap, Layer, std::shared_ptr<Bootstrap>>(m, "Bootstrap").def(py::init<>());
  py::class_<Residual, Layer, std::shared_ptr<Residual>>(m, "Residual")
      .def(py::init<std::vector<LayerPtr>>(), py::arg("body"));
  py::class_<LayerNorm, Layer, std::shared_ptr<LayerNorm>>(m, "LayerNorm")
      .def(py::init<int, int, Vec, Vec, double, double, double, uint32_t, uint32_t>(), py::arg("tokens"),
           py::arg("dim"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5, py::arg("var_min") = 0.1,
           py::arg("var_max") = 1.0, py::arg("degree") = 15, py::arg("newton") = 1,
           "Per-token layer norm; 1/sqrt(var + eps) is a degree-`degree` Chebyshev series on "
           "[var_min, var_max] refined by `newton` Newton steps.")
      .def("inv_sqrt", &LayerNorm::inv_sqrt, py::arg("var"));
  py::class_<SoftmaxOptions>(m, "SoftmaxOptions")
      .def(py::init<>())
      .def_readwrite("lo", &SoftmaxOptions::lo)
      .def_readwrite("hi", &SoftmaxOptions::hi)
      .def_readwrite("exp_degree", &SoftmaxOptions::exp_degree)
      .def_readwrite("inv_degree", &SoftmaxOptions::inv_degree)
      .def_readwrite("newton", &SoftmaxOptions::newton)
      .def_readwrite("subtract_max", &SoftmaxOptions::subtract_max)
      .def_readwrite("sum_lo", &SoftmaxOptions::sum_lo)
      .def_readwrite("sum_hi", &SoftmaxOptions::sum_hi);
  py::class_<Attention, Layer, std::shared_ptr<Attention>>(m, "Attention")
      .def(py::init<int, int, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, int>(), py::arg("tokens"), py::arg("dim"),
           py::arg("wq"), py::arg("wk"), py::arg("wv"), py::arg("wo"), py::arg("bq") = Vec{}, py::arg("bk") = Vec{},
           py::arg("bv") = Vec{}, py::arg("bo") = Vec{}, py::arg("score") = Vec{0.0, 0.0, 1.0}, py::arg("heads") = 1,
           "Self-attention with `heads` heads of width dim/heads; score is (c0, c1, c2) shared by all heads "
           "or 3*heads values, one polynomial per head.")
      .def_property_readonly("heads", &Attention::heads)
      .def("set_softmax", &Attention::set_softmax, py::arg("options"),
           "Softmax scores (polynomial exp and 1/x, optionally with the row maximum subtracted through "
           "scheme switching) instead of the quadratic score function; compile the network again afterwards.")
      .def_property_readonly("softmax", &Attention::softmax);

  py::class_<Network>(m, "Network")
      .def(py::init<Shape>(), py::arg("input_shape"))
      .def_static("load", &Network::load, py::arg("path"))
      .def("save", &Network::save, py::arg("path"))
      .def(
          "add", [](Network& n, LayerPtr l) -> Network& { return n.add(std::move(l)); }, py::arg("layer"),
          py::return_value_policy::reference_internal)
      .def("fuse_linear", &Network::fuse_linear)
      .def("use_exact_relu", &Network::use_exact_relu, py::arg("range") = 0.0,
           "Replace Chebyshev relu activations by scheme-switching ExactReLU; returns how many.")
      .def("place_bootstraps", &Network::place_bootstraps, py::arg("budget"),
           py::arg("calibration") = std::vector<std::vector<double>>{}, py::arg("headroom") = 1.5,
           "Insert bootstraps so no stretch exceeds `budget` levels; compile() again afterwards.")
      .def("fold_activation_intervals", &Network::fold_activation_intervals)
      .def_static("max_bootstrap_budget", &Network::max_bootstrap_budget, py::arg("base"),
                  py::arg("max_ring") = 1u << 16)
      .def_property_readonly("depth_between_bootstraps", &Network::depth_between_bootstraps)
      .def_property_readonly("uses_bootstrapping", &Network::uses_bootstrapping)
      .def_property_readonly("uses_scheme_switching", &Network::uses_scheme_switching)
      .def_property_readonly("internal_bootstraps", &Network::internal_bootstraps,
                             "Bootstraps run inside self-bootstrapping layers per inference.")
      .def("compile", &Network::compile, py::arg("slots") = 0, py::call_guard<py::gil_scoped_release>())
      .def("set_gapped_layouts", &Network::set_gapped_layouts)
      .def_property(
          "packing",
          [](const Network& n) {
            switch (n.packing()) {
              case Packing::kCompact: return std::string("compact");
              case Packing::kGapped: return std::string("gapped");
              default: return std::string("multiplexed");
            }
          },
          [](Network& n, const std::string& p) {
            if (p == "compact") n.set_packing(Packing::kCompact);
            else if (p == "gapped") n.set_packing(Packing::kGapped);
            else if (p == "multiplexed") n.set_packing(Packing::kMultiplexed);
            else throw std::invalid_argument("packing must be compact, gapped or multiplexed");
          },
          "Layout of strided conv/pool outputs: 'gapped' (default, fewest diagonals), 'compact' (fewest "
          "slots) or 'multiplexed' (compact slot count, few diagonals; Lee et al. ICML'22). Compile again "
          "afterwards.")
      .def("set_cache_plaintexts", &Network::set_cache_plaintexts)
      .def_property_readonly("depth", &Network::depth)
      .def_property_readonly("slots", &Network::slots)
      .def_property_readonly("input_shape", &Network::input_shape)
      .def_property_readonly("output_shape", &Network::output_shape)
      .def_property_readonly("layers", &Network::layers)
      .def("required_slots", &Network::required_slots)
      .def("rotation_indices", &Network::rotation_indices, py::arg("ladder") = false)
      .def("context_options", &Network::context_options, py::arg("secure") = true)
      .def("input_level", &Network::input_level, py::arg("ctx"))
      .def("summary", &Network::summary)
      .def("forward_plain", &Network::forward_plain, py::arg("x"), py::call_guard<py::gil_scoped_release>())
      .def(
          "forward",
          [](const Network& n, const FHEContext& ctx, const Ciphertext& x, const py::object& hook) {
            Network::LayerHook h;
            if (!hook.is_none())
              h = [&](size_t i, const Layer& l, const Ctxt& c, double secs) {
                py::gil_scoped_acquire acquire;
                hook(i, l.describe(), Ciphertext{c}, secs);
              };
            py::gil_scoped_release release;
            return Ciphertext{n.forward(ctx, x.ct, h)};
          },
          py::arg("ctx"), py::arg("x"), py::arg("hook") = py::none(),
          "hook(index, description, ciphertext, seconds) is called after every layer.")
      .def(
          "decrypt_output", [](const Network& n, const FHEContext& ctx, const Ciphertext& y) { return n.decrypt_output(ctx, y.ct); },
          py::arg("ctx"), py::arg("y"))
      .def(
          "decrypt_layer",
          [](const Network& n, const FHEContext& ctx, const Ciphertext& y, size_t i) { return n.decrypt_layer(ctx, y.ct, i); },
          py::arg("ctx"), py::arg("y"), py::arg("layer"), "Decrypts the output of layer i (e.g. from a forward hook).")
      .def("encrypt_weights", &Network::encrypt_weights, py::arg("ctx"), py::arg("drop_plaintext") = true,
           py::call_guard<py::gil_scoped_release>());

  py::class_<Loss> loss(m, "Loss", "Training loss: Loss.mse() or Loss.cross_entropy(SoftmaxOptions) (see train.h).");
  py::enum_<Loss::Kind>(loss, "Kind").value("MSE", Loss::kMSE).value("CROSS_ENTROPY", Loss::kCrossEntropy);
  loss.def(py::init<>())
      .def_readwrite("kind", &Loss::kind)
      .def_readwrite("softmax", &Loss::softmax)
      .def_static("mse", &Loss::mse)
      .def_static("cross_entropy", &Loss::cross_entropy, py::arg("softmax") = SoftmaxOptions{},
                  "Softmax cross-entropy against one-hot targets; the softmax uses the polynomial exp and 1/x "
                  "(optionally with the maximum subtracted through scheme switching).");
  m.def("softmax_plain", &softmax_plain, py::arg("loss"), py::arg("y"),
        "The softmax the cross-entropy Trainer computes under encryption.");
  m.def("cross_entropy_plain", &cross_entropy_plain, py::arg("y"), py::arg("target"),
        "Exact -sum t log softmax(y).");

  py::class_<Trainer>(m, "Trainer", "Encrypted SGD training (see include/fheml/train.h).")
      .def(py::init<Network&, std::vector<size_t>, Loss>(), py::arg("net"), py::arg("frozen") = std::vector<size_t>{},
           py::arg("loss") = Loss{}, py::keep_alive<1, 2>())
      .def_property_readonly("loss", &Trainer::loss)
      .def_property_readonly("step_depth", &Trainer::step_depth)
      .def_property_readonly("unbounded_step_depth", &Trainer::unbounded_step_depth)
      .def("set_level_budget", &Trainer::set_level_budget, py::arg("levels"), py::arg("range") = 1.0,
           "Bootstrap activations and gradients so that no ciphertext of a step goes deeper than `levels`.")
      .def_property_readonly("level_budget", &Trainer::level_budget)
      .def_property_readonly("bootstrap_range", &Trainer::bootstrap_range)
      .def_property_readonly("min_level_budget", &Trainer::min_level_budget)
      .def_property_readonly("sample_bootstraps", &Trainer::sample_bootstraps)
      .def_property_readonly("weight_ciphertexts", &Trainer::weight_ciphertexts)
      .def_property_readonly("first_trained_layer", &Trainer::first_trained_layer)
      .def_property_readonly("weight_level", &Trainer::weight_level)
      .def("rotation_indices", &Trainer::rotation_indices, py::arg("ladder") = false)
      .def("context_options", &Trainer::context_options, py::arg("secure") = true, py::arg("bootstrapping") = true,
           py::arg("updates") = 1, py::arg("rotation_window") = 0)
      .def("encrypt_weights", &Trainer::encrypt_weights, py::arg("ctx"), py::call_guard<py::gil_scoped_release>())
      .def(
          "encrypt_input",
          [](const Trainer& t, const FHEContext& c, const Vec& x) { return Ciphertext{t.encrypt_input(c, x)}; },
          py::arg("ctx"), py::arg("x"))
      .def(
          "encrypt_target",
          [](const Trainer& t, const FHEContext& c, const Vec& y) { return Ciphertext{t.encrypt_target(c, y)}; },
          py::arg("ctx"), py::arg("target"))
      .def(
          "accumulate",
          [](Trainer& t, const FHEContext& c, const Ciphertext& x, const Ciphertext& y) {
            py::gil_scoped_release release;
            return Ciphertext{t.accumulate(c, x.ct, y.ct)};
          },
          py::arg("ctx"), py::arg("x"), py::arg("target"))
      .def("update", &Trainer::update, py::arg("ctx"), py::arg("lr"), py::call_guard<py::gil_scoped_release>())
      .def(
          "forward",
          [](Trainer& t, const FHEContext& c, const Ciphertext& x) {
            py::gil_scoped_release release;
            return Ciphertext{t.forward(c, x.ct)};
          },
          py::arg("ctx"), py::arg("x"))
      .def(
          "decrypt_output", [](const Trainer& t, const FHEContext& c, const Ciphertext& y) { return t.decrypt_output(c, y.ct); },
          py::arg("ctx"), py::arg("y"))
      .def("decrypt_weights", &Trainer::decrypt_weights, py::arg("ctx"), py::call_guard<py::gil_scoped_release>())
      .def("summary", &Trainer::summary);

  m.def("gpu_backend_built", &gpu_backend_built, "True when built with FIDESlib (-DFHEML_GPU=ON).");
  m.def("gpu_device_count", &gpu_device_count, "Visible CUDA devices (0 without the GPU backend).");
  m.def("train_plain", &train_plain, py::arg("net"), py::arg("xs"), py::arg("targets"), py::arg("lr"),
        py::arg("frozen") = std::vector<size_t>{}, py::arg("loss") = Loss{},
        "Plaintext reference of Trainer: one SGD update on a batch.",
        py::call_guard<py::gil_scoped_release>());
}
