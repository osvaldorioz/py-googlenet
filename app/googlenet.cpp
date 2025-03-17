#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Función para realizar una convolución 2D
Eigen::MatrixXd conv2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel) {
    int kernel_size = kernel.rows();
    int output_size = input.rows() - kernel_size + 1;
    Eigen::MatrixXd output(output_size, output_size);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            output(i, j) = (input.block(i, j, kernel_size, kernel_size).cwiseProduct(kernel)).sum();
        }
    }

    return output;
}

// Función para realizar max pooling
Eigen::MatrixXd max_pool2d(const Eigen::MatrixXd& input, int pool_size) {
    int output_size = input.rows() / pool_size;
    Eigen::MatrixXd output(output_size, output_size);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            output(i, j) = input.block(i * pool_size, j * pool_size, pool_size, pool_size).maxCoeff();
        }
    }

    return output;
}

// Clase para la red GoogleNet inspirada en LeNet
class GoogleNet {
public:
    GoogleNet() {
        // Inicialización de kernels (filtros) y pesos
        kernel1 = Eigen::MatrixXd::Random(5, 5);  // Kernel de 5x5 para la primera capa convolucional
        kernel2 = Eigen::MatrixXd::Random(5, 5);  // Kernel de 5x5 para la segunda capa convolucional
        weights_fc1 = Eigen::MatrixXd::Random(320, 50);  // Pesos para la primera capa fully connected
        weights_fc2 = Eigen::MatrixXd::Random(50, 10);   // Pesos para la segunda capa fully connected
    }

    // Función de forward pass que acepta un array de NumPy
    py::array_t<double> forward(py::array_t<double>& input) {
        // Convertir el array de NumPy a Eigen::MatrixXd
        py::buffer_info buf = input.request();
        double* ptr = static_cast<double*>(buf.ptr);
        Eigen::Map<Eigen::MatrixXd> input_matrix(ptr, buf.shape[0], buf.shape[1]);

        // Primera capa convolucional + ReLU + MaxPool
        Eigen::MatrixXd x = conv2d(input_matrix, kernel1).cwiseMax(0);  // ReLU
        x = max_pool2d(x, 2);  // MaxPool

        // Segunda capa convolucional + ReLU + MaxPool
        x = conv2d(x, kernel2).cwiseMax(0);  // ReLU
        x = max_pool2d(x, 2);  // MaxPool

        // Aplanar
        Eigen::VectorXd flattened = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());

        // Primera capa fully connected + ReLU
        Eigen::VectorXd fc1_output = (weights_fc1.transpose() * flattened).cwiseMax(0);

        // Segunda capa fully connected
        Eigen::VectorXd fc2_output = weights_fc2.transpose() * fc1_output;

        // Convertir la salida de Eigen::VectorXd a un array de NumPy
        return py::array_t<double>(
            {fc2_output.size()},  // Shape
            {sizeof(double)},     // Stride
            fc2_output.data()     // Puntero a los datos
        );
    }

private:
    Eigen::MatrixXd kernel1, kernel2;  // Kernels para las capas convolucionales
    Eigen::MatrixXd weights_fc1, weights_fc2;  // Pesos para las capas fully connected
};

// Enlazar la clase GoogleNet a Python con Pybind11
PYBIND11_MODULE(googlenet, m) {
    py::class_<GoogleNet>(m, "GoogleNet")
        .def(py::init<>())
        .def("forward", &GoogleNet::forward);
}