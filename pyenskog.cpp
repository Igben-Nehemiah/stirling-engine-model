#include "enskog.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pyenskog, m) {
  py::class_<Enskog> enskog(m, "Enskog");

  enskog.def(py::init<double, int, size_t, bool, double, bool>())
    .def("addSpecies", &Enskog::addSpecies)
    .def("init", &Enskog::init)
    .def("pressure", &Enskog::pressure)
    .def("packing_fraction", &Enskog::packing_fraction)
    .def("Z", &Enskog::Z)
    .def("Lab", &Enskog::Lab)
    .def("Lau", &Enskog::Lau)
    .def("Lua", &Enskog::Lua)
    .def("Luu", &Enskog::Luu)
    .def("shearViscosity", &Enskog::shearViscosity)
    .def("bulkViscosity", &Enskog::bulkViscosity)
    .def("chemicalPotential", &Enskog::chemicalPotential)
    .def("gr", &Enskog::gr)
    .def("validation", &Enskog::validation)
    .def("normalise_concs", &Enskog::normalise_concs)
    .def("generate_gr_array", &Enskog::generate_gr_array)
    .def("Solvea", &Enskog::Solvea)
    .def("Solveb", &Enskog::Solveb)
    .def("Solved", &Enskog::Solved)
    .def("Solveh", &Enskog::Solveh)
    .def("Bpqdijl", &Enskog::Bpqdijl)
    .def("Bpqddijl", &Enskog::Bpqddijl)
    .def("Bpqdij0", &Enskog::Bpqdij0)
    .def("Bpqddij0", &Enskog::Bpqddij0)
    .def("Sd1o2", &Enskog::Sd1o2)
    .def("Sdd1o2", &Enskog::Sdd1o2)
    .def("Sd3o2", &Enskog::Sd3o2)
    .def("Sdd3o2", &Enskog::Sdd3o2)
    .def("Sd5o2", &Enskog::Sd5o2)
    .def("Sdd5o2", &Enskog::Sdd5o2)
    ;

  py::enum_<Enskog::EOStype>(m, "EOStype")
    .value("BOLTZMANN", Enskog::BOLTZMANN)
    .value("BMCSL", Enskog::BMCSL)
    .value("VS", Enskog::VS)
    .value("HEYES", Enskog::HEYES)
    ;
}
