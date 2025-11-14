#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
using namespace std;

#define BLOCK_SIZE 256 //numero de hilos por cada bloque, tamaño en pocas palabras


__host__ __device__ inline double sigmoid(double x) {
    return static_cast<double>(1.0) / (static_cast<double>(1.0) + exp(-x));
}


__global__ void forward_kernel(const double* __restrict__ W,
                               const double* __restrict__ b,
                               const double* __restrict__ input,
                               double* __restrict__ output,
                               int input_size, int output_size) {
    int i = blockIdx.x;    // neurona (salida)
    int tid = threadIdx.x;

    if (i >= output_size) return;

    __shared__ double sum[BLOCK_SIZE];
    sum[tid] = 0.0;

    for (int j = tid; j < input_size; j += blockDim.x) {
        sum[tid] += W[i * input_size + j] * input[j];
    }
    __syncthreads();

    if (tid == 0) {
        double total = b[i];
        // Sumar solo la cantidad double de hilos del bloque
        for (int k = 0; k < blockDim.x; ++k) total += sum[k];
        output[i] = sigmoid(total);
    }
}

__global__ void backward_kernel(double* __restrict__ W,
                                       double* __restrict__ b,
                                       const double* __restrict__ grad_out, // ya incluye derivada de sigmoid
                                       const double* __restrict__ input,
                                       int input_size, int output_size,
                                       double lr) {
    int i = blockIdx.x;   
    int tid = threadIdx.x;

    if (i >= output_size) return;
    for (int col = tid; col < input_size; col += blockDim.x) {
        //atomicAdd(&W[i * input_size + col], lr * grad);
        W[i * input_size + col] += lr * grad_out[i] * input[col];
    }

    // Sesgo: un solo hilo por bloque
    if (tid == 0) {
        //atomicAdd(&b[i], lr * grad_out[i]);
        b[i] += lr * grad_out[i];
    }
}

struct CapaDensa {
    int input_size, output_size;
    vector<double> h_W, h_b;
    double *d_W, *d_b, *d_input, *d_output;

    CapaDensa(int in_size, int out_size) {
        input_size = in_size;
        output_size = out_size;

        h_W.resize(out_size * in_size);
        h_b.resize(out_size, 0.0);

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, 0.01);
        for (auto &w : h_W) w = dist(gen);

        cudaMalloc(&d_W, sizeof(double) * out_size * in_size);
        cudaMalloc(&d_b, sizeof(double) * out_size);
        cudaMalloc(&d_input, sizeof(double) * in_size);
        cudaMalloc(&d_output, sizeof(double) * out_size);

        cudaMemcpy(d_W, h_W.data(), sizeof(double) * out_size * in_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), sizeof(double) * out_size, cudaMemcpyHostToDevice);
    }

    ~CapaDensa() {
        cudaFree(d_W);
        cudaFree(d_b);
        cudaFree(d_input);
        cudaFree(d_output);
    }

    vector<double> forward(const vector<double>& input) {
        cudaMemcpy(d_input, input.data(), sizeof(double) * input_size, cudaMemcpyHostToDevice);
        forward_kernel<<<output_size, BLOCK_SIZE>>>(d_W, d_b, d_input, d_output,
                                                    input_size, output_size);
        cudaGetLastError();
        cudaDeviceSynchronize();
        vector<double> out(output_size);
        cudaMemcpy(out.data(), d_output, sizeof(double) * output_size, cudaMemcpyDeviceToHost);
        return out;
    }

    void backward(const vector<double>& input, const vector<double>& grad_out, double lr) {
        double *d_grad_out;
        cudaMalloc(&d_grad_out, sizeof(double) * output_size);
        cudaMemcpy(d_grad_out, grad_out.data(), sizeof(double) * output_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input, input.data(), sizeof(double) * input_size, cudaMemcpyHostToDevice);

        backward_kernel<<<output_size, BLOCK_SIZE>>>(d_W, d_b, d_grad_out, d_input,
                                                            input_size, output_size, lr);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaFree(d_grad_out);
    }
};

struct SimpleMLP {
    CapaDensa output;
    double lr;

    SimpleMLP(int input_size, int output_size, double lr_)
        : output(input_size, output_size), lr(lr_) {}

    vector<double> forward(const vector<double>& x) {
        return output.forward(x);
    }

    void backward(const vector<double>& x, const vector<double>& target, const vector<double>& pred) {
        vector<double> grad_out(pred.size());
        for (size_t i = 0; i < pred.size(); i++) {
            grad_out[i] = (target[i] - pred[i]) * (pred[i] * (1.0 - pred[i]));
        }
        output.backward(x, grad_out, lr);
    }

    double compute_loss(const vector<double>& pred, int label) {
        double loss = 0.0;
        for (int i = 0; i < (int)pred.size(); i++) {
            double t = (i == label ? 1.0 : 0.0);
            double diff = pred[i] - t;
            loss += diff * diff;
        }
        return loss / pred.size();
    }

    int predict(const vector<double>& x) {
        vector<double> out = forward(x);
        return int(max_element(out.begin(), out.end()) - out.begin());
    }

    double evaluate(const vector<vector<double>>& X, const vector<int>& y) {
        int correct = 0;
        for (int i = 0; i < (int)X.size(); i++) {
            if (predict(X[i]) == y[i]) correct++;
        }
        return (double)correct / X.size() * 100.0;
    }

    void fit(const vector<vector<double>>& X, const vector<int>& y,
             int epochs, int batch_size,
             const vector<vector<double>>& X_val = {},
             const vector<int>& y_val = {}) {

        int n = (int)X.size();
        vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);

        ofstream log_file("loss_history.txt");
        log_file << "epoch,train_loss,val_loss\n";

        int num_batches = (n + batch_size - 1) / batch_size;

        for (int epoch = 0; epoch < epochs; epoch++) {
            random_shuffle(indices.begin(), indices.end());
            double total_loss = 0.0;

            for (int start = 0; start < n; start += batch_size) {
                int end = min(start + batch_size, n);
                double batch_loss = 0.0;

                for (int i = start; i < end; i++) {
                    int idx = indices[i];
                    auto pred = forward(X[idx]);

                    vector<double> target(10, 0.0);
                    target[y[idx]] = 1.0;

                    batch_loss += compute_loss(pred, y[idx]);
                    backward(X[idx], target, pred);
                }
                total_loss += batch_loss / (end - start);

                cudaDeviceSynchronize();
            }

            double avg_train_loss = total_loss / num_batches;
            double val_loss = 0.0;

            if (!X_val.empty()) {
                for (int i = 0; i < (int)X_val.size(); i++) {
                    auto pred_val = forward(X_val[i]);
                    val_loss += compute_loss(pred_val, y_val[i]);
                }
                val_loss /= X_val.size();
            }

            cudaDeviceSynchronize();

            cout << "Epoch " << epoch + 1
                 << " - Train Loss: " << fixed << setprecision(5) << avg_train_loss;
            if (!X_val.empty())
                cout << " - Val Loss: " << val_loss;
            cout << endl;

            log_file << epoch + 1 << "," << avg_train_loss << ",";
            if (!X_val.empty()) log_file << val_loss;
            log_file << "\n";
        }

        log_file.close();
    }
};


void load_cifar10_batch(const string& filename, vector<vector<double>>& images, vector<int>& labels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("No se pudo abrir " + filename);

    const int num_images = 10000;
    const int image_size = 3072; // 32x32x3
    for (int i = 0; i < num_images; i++) {
        unsigned char label;
        file.read((char*)&label, 1);
        labels.push_back((int)label);

        vector<double> img(image_size);
        unsigned char buffer[image_size];
        file.read((char*)buffer, image_size);

        for (int j = 0; j < image_size; j++)
            img[j] = (buffer[j] / 255.0 - 0.5) * 2.0;

        images.push_back(std::move(img));
    }
}

int main() {
    vector<vector<double>> train_images, test_images;
    vector<int> train_labels, test_labels;

    for (int i = 1; i <= 5; i++) {
        string fname = "data_batch_" + to_string(i) + ".bin";
        load_cifar10_batch(fname, train_images, train_labels);
    }
    load_cifar10_batch("test_batch.bin", test_images, test_labels);

    cout << "Cargadas " << train_images.size() << " train y " << test_images.size() << " test" << endl;

    int input_size = 32 * 32 * 3;
    int output_size = 10;
    int epochs = 100;
    int minibatch_size = 250;
    double learning_rate = 0.01;

    SimpleMLP model(input_size, output_size, learning_rate);

    model.fit(train_images, train_labels, epochs, minibatch_size, test_images, test_labels);

    double train_acc = model.evaluate(train_images, train_labels);
    double test_acc = model.evaluate(test_images, test_labels);

    cout << "Train acc: " << train_acc << "%" << endl;
    cout << "Test acc: " << test_acc << "%" << endl;

    return 0;

}


