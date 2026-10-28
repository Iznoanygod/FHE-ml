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
      .def_property_readonly("on_gpu", &FHEContext::on_gpu)
      .def("describe", &FHEContext::describe)
      .def_static("ring_dim_for", &FHEContext::ring_dim_for, py::arg("options"));

  py::class_<Layer, LayerPtr>(m, "Layer")
      .def_property_readonly("type", &Layer::type)
      .def("describe", &Layer::describe)
      .def_property_readonly("depth", &Layer::depth)
      .def_property_readonly("in_shape", &Layer::in_shape)
      .def_property_readonly("out_shape", &Layer::out_shape)
      .def("forward_plain", &Layer::forward_plain);

  py::class_<LinearLayer, Layer, std::shared_ptr<LinearLayer>>(m, "LinearLayer")
      .def("params", &LinearLayer::params, "Trainable values: weights, then biases.")
      .def("set_params", &LinearLayer::set_params, py::arg("params"))
      .def_property_readonly("trainable", &LinearLayer::trainable);
  py::class_<Conv2d, LinearLayer, std::shared_ptr<Conv2d>>(m, "Conv2d")
      .def(py::init<int, int, int, int, int, int, Vec, Vec>(), py::arg("in_channels"), py::arg("out_channels"),
           py::arg("kh"), py::arg("kw"), py::arg("stride"), py::arg("pad"), py::arg("weights"),
           py::arg("bias") = Vec{});
  py::class_<Dense, LinearLayer, std::shared_ptr<Dense>>(m, "Dense")
      .def(py::init<int, int, Vec, Vec>(), py::arg("inputs"), py::arg("outputs"), py::arg("weights"),
           py::arg("bias") = Vec{});
  py::class_<AvgPool2d, LinearLayer, std::shared_ptr<AvgPool2d>>(m, "AvgPool2d")
      .def(py::init<int, int>(), py::arg("k"), py::arg("stride"));
  py::class_<ChannelAffine, LinearLayer, std::shared_ptr<ChannelAffine>>(m, "ChannelAffine")
      .def(py::init<Vec, Vec>(), py::arg("scale"), py::arg("shift"));
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
           py::arg("degree"));
  py::class_<Bootstrap, Layer, std::shared_ptr<Bootstrap>>(m, "Bootstrap").def(py::init<>());
  py::class_<Residual, Layer, std::shared_ptr<Residual>>(m, "Residual")
      .def(py::init<std::vector<LayerPtr>>(), py::arg("body"));
  py::class_<Attention, Layer, std::shared_ptr<Attention>>(m, "Attention")
      .def(py::init<int, int, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec>(), py::arg("tokens"), py::arg("dim"),
           py::arg("wq"), py::arg("wk"), py::arg("wv"), py::arg("wo"), py::arg("bq") = Vec{}, py::arg("bk") = Vec{},
           py::arg("bv") = Vec{}, py::arg("bo") = Vec{}, py::arg("score") = Vec{0.0, 0.0, 1.0});

  py::class_<Network>(m, "Network")
      .def(py::init<Shape>(), py::arg("input_shape"))
      .def_static("load", &Network::load, py::arg("path"))
      .def("save", &Network::save, py::arg("path"))
      .def(
          "add", [](Network& n, LayerPtr l) -> Network& { return n.add(std::move(l)); }, py::arg("layer"),
          py::return_value_policy::reference_internal)
      .def("fuse_linear", &Network::fuse_linear)
      .def("compile", &Network::compile, py::arg("slots") = 0, py::call_guard<py::gil_scoped_release>())
      .def("set_gapped_layouts", &Network::set_gapped_layouts)
      .def("set_cache_plaintexts", &Network::set_cache_plaintexts)
      .def_property_readonly("depth", &Network::depth)
      .def_property_readonly("slots", &Network::slots)
      .def_property_readonly("input_shape", &Network::input_shape)
      .def_property_readonly("output_shape", &Network::output_shape)
      .def_property_readonly("layers", &Network::layers)
      .def("required_slots", &Network::required_slots)
      .def("rotation_indices", &Network::rotation_indices)
      .def("context_options", &Network::context_options, py::arg("secure") = true)
      .def("input_level", &Network::input_level, py::arg("ctx"))
      .def("summary", &Network::summary)
      .def("forward_plain", &Network::forward_plain, py::arg("x"), py::call_guard<py::gil_scoped_release>())
      .def(
          "forward",
          [](const Network& n, const FHEContext& ctx, const Ciphertext& x) {
            py::gil_scoped_release release;
            return Ciphertext{n.forward(ctx, x.ct)};
          },
          py::arg("ctx"), py::arg("x"))
      .def(
          "decrypt_output", [](const Network& n, const FHEContext& ctx, const Ciphertext& y) { return n.decrypt_output(ctx, y.ct); },
          py::arg("ctx"), py::arg("y"))
      .def("encrypt_weights", &Network::encrypt_weights, py::arg("ctx"), py::arg("drop_plaintext") = true,
           py::call_guard<py::gil_scoped_release>());

  py::class_<Trainer>(m, "Trainer", "Encrypted SGD training (see include/fheml/train.h).")
      .def(py::init<Network&, std::vector<size_t>>(), py::arg("net"), py::arg("frozen") = std::vector<size_t>{},
           py::keep_alive<1, 2>())
      .def_property_readonly("step_depth", &Trainer::step_depth)
      .def_property_readonly("weight_ciphertexts", &Trainer::weight_ciphertexts)
      .def_property_readonly("first_trained_layer", &Trainer::first_trained_layer)
      .def_property_readonly("weight_level", &Trainer::weight_level)
      .def("rotation_indices", &Trainer::rotation_indices)
      .def("context_options", &Trainer::context_options, py::arg("secure") = true, py::arg("bootstrapping") = true,
           py::arg("updates") = 1)
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
        py::arg("frozen") = std::vector<size_t>{}, "Plaintext reference of Trainer: one SGD update on a batch.",
        py::call_guard<py::gil_scoped_release>());
}
