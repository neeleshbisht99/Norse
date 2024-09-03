#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "norse.h"

namespace py = pybind11;

PYBIND11_MODULE(NorseNet, m) {
    m.doc() = "A lightweight and easy-to-use deep learning framework with neural architecture search.";

    py::class_<NorseNet::Norse>(m, "Norse")
            .def(py::init<uint32_t, uint32_t, uint32_t, const std::string &>(), py::arg("capacity"),
                 py::arg("inputDim"), py::arg("outputDim"), py::arg("mode"))
            .def("preheat", &NorseNet::Norse::preheat, py::arg("nodes"), py::arg("connections"), py::arg("depths"),
                 "Preheating process of the neural network cluster.")
            .def("evolve", &NorseNet::Norse::evolve, py::arg("input"), py::arg("desire"),
                 "Evolve the neural network cluster.")
            .def("fit", &NorseNet::Norse::fit, py::arg("input"), py::arg("desire"),
                 "Fit all neural networks in the neural network cluster.")
            .def("enrich", &NorseNet::Norse::enrich, "Keep only the best one in the neural network cluster.")
            .def("compute", &NorseNet::Norse::compute, py::arg("input"),
                 "Forward Computing of the best individual.")
            .def("computeBatch", &NorseNet::Norse::computeBatch, py::arg("input"),
                 "Parallel forward Computing of the best individual.")
            .def("resize", &NorseNet::Norse::resize, py::arg("capacity"),
                 "Resize the capacity of the neural network cluster.")
            .def("openMemLimit", &NorseNet::Norse::openMemLimit, py::arg("size"), "Turn on memory-limit.")
            .def("closeMemLimit", &NorseNet::Norse::closeMemLimit, "Turn off memory-limit.")
            .def("saveModel", &NorseNet::Norse::saveModel, py::arg("path"), "Export the current trained model.")
            .def("setMutateArgs", &NorseNet::Norse::setMutateArgs, py::arg("p"), "Set p1 to p4 in the class Args.")
            .def("setMutateOdds", &NorseNet::Norse::setMutateOdds, py::arg("odds"), "Set the odds of mutating.")
            .def("setCpuCores", &NorseNet::Norse::setCpuCores, py::arg("num"),
                 "Set the number of worker threads to train the model.")
            .def("setLR", &NorseNet::Norse::setLR, py::arg("lr"), "Set the learning rate.")
            .def("getSize", &NorseNet::Norse::getSize, "Returns the size of the best individual.")
            .def("getInputDim", &NorseNet::Norse::getInputDim, "Returns the input dimension.")
            .def("getOutputDim", &NorseNet::Norse::getOutputDim, "Returns the output dimension.")
            .def("getCapacity", &NorseNet::Norse::getCapacity, "Returns capacity of Norse.")
            .def("getLoss", &NorseNet::Norse::getLoss, "Returns the loss.")
            .def("getMode", &NorseNet::Norse::getMode, "Returns mode of Norse (classify or predict).")
            .def("getLayers", &NorseNet::Norse::getLayers,
                 "Returns the number of nodes in each layer of the neural network.")
            .def("getHiddenLayer", &NorseNet::Norse::getHiddenLayer, py::arg("pos"),
                 "Get a vector of nodes in a specific layer of the best individual.")
            .def("version", &NorseNet::Norse::version, "Returns version information and copyright information.");

    m.def("loadModel", &NorseNet::loadModel, py::arg("path"), py::arg("capacity"),
          "Import the pre-trained model.");
    m.def("loadViaString", &NorseNet::loadViaString, py::arg("model"), py::arg("capacity"),
          "Import the pre-trained model via string.");
}