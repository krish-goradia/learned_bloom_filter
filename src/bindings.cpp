#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include "../include/BloomFilter.h"

namespace py = pybind11;

PYBIND11_MODULE(bloom,m) {
    py::class_<BFStats>(m,"BFStats")
        .def_readonly("memory_bits",&BFStats::memory_bits)
        .def_readonly("bits_per_element",&BFStats::bits_per_element)
        .def_readonly("avg_latency_ns",&BFStats::avg_latency_ns)
        .def_readonly("throughput_qps",&BFStats::throughput_qps);


    py::class_<BloomFilter>(m,"BloomFilter")
        .def(py::init<size_t,size_t>(),py::arg("m"),py::arg("k"))
        .def(py::init<size_t,double>(),py::arg("n"),py::arg("fpr"))
        .def("insert",&BloomFilter::insert)
        .def("contains",&BloomFilter::contains)
        .def("insert_batch",&BloomFilter::insert_batch)
        .def("clear",&BloomFilter::clear)
        .def("getTheoreticalSize",&BloomFilter::getTheoreticalBits)
        .def("query_batch",&BloomFilter::query_batch)
        .def("getMemoryBits",&BloomFilter::getMemoryBits)
        .def("getMemoryBytes",&BloomFilter::getMemoryBytes)
        .def("getBitsPerElement",&BloomFilter::getBitsPerElement)
        .def("benchmark",&BloomFilter::benchmark);

}