#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include "../include/BloomFilter.h"

namespace py = pybind11;

PYBIND11_MODULE(bloom,m) {
    py::class_<BloomFilter>(m,"BloomFilter")
        .def(py::init<size_t,size_t>(),py::arg("m"),py::arg("k"))
        .def(py::init<size_t,double>(),py::arg("n"),py::arg("fpr"))
        .def("insert",&BloomFilter::insert)
        .def("contains",&BloomFilter::contains)
        .def("insert_batch",&BloomFilter::insert_batch)
        .def("clear",&BloomFilter::clear)
        .def("size",&BloomFilter::getSize)
        .def("query_batch",&BloomFilter::query_batch);

}