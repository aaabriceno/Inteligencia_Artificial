#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>
using namespace std;

#define BLOCK_SIZE 256

__host__ __device__ inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// forward: 1 bloque por neurona de salida, BLOCK_SIZE hilos por bloque
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

    // Acumulación en stride por hilo
    for (int j = tid; j < input_size; j += blockDim.x) {
        sum[tid] += W[i * input_size + j] * input[j];
    }
    __syncthreads();

    if (tid == 0) {
        double total = b[i];
        // Sumar las contribuciones parciales
        for (int k = 0; k < blockDim.x; ++k) total += sum[k];
        output[i] = sigmoid(total);
    }
}

// calcula grad_input = W^T * grad_out
// bloques = input_size, cada bloque reduce sobre output_size
__global__ void matvec_transpose_kernel(const double* __restrict__ W,
                                        const double* __restrict__ grad_out,
                                        double* __restrict__ grad_in,
                                        int input_size, int output_size) {
    int j = blockIdx.x; // neurona de la capa anterior (entrada para esta capa)
    int tid = threadIdx.x;
    if (j >= input_size) return;

    __shared__ double sum[BLOCK_SIZE];
    sum[tid] = 0.0;

    for (int i = tid; i < output_size; i += blockDim.x) {
        sum[tid] += W[i * input_size + j] * grad_out[i];
    }
    __syncthreads();

    if (tid == 0) {
        double s = 0.0;
        for (int k = 0; k < blockDim.x; ++k) s += sum[k];
        grad_in[j] = s;
    }
}

// actualizar W y b: un bloque por neurona de salida
__global__ void update_weights_kernel(double* __restrict__ W,
                                      double* __restrict__ b,
                                      const double* __restrict__ grad_out, // ya incluye derivada de activación
                                      const double* __restrict__ input,
                                      int input_size, int output_size,
                                      double lr) {
    int i = blockIdx.x;   // neurona salida
    int tid = threadIdx.x;

    if (i >= output_size) return;

    for (int col = tid; col < input_size; col += blockDim.x) {
        W[i * input_size + col] += lr * grad_out[i] * input[col];
    }

    if (tid == 0) {
        b[i] += lr * grad_out[i];
    }
}

struct DenseLayer {
    int input_size, output_size;
    vector<double> h_W, h_b;
    double *d_W, *d_b, *d_input, *d_output;

    DenseLayer() : input_size(0), output_size(0),
                   d_W(nullptr), d_b(nullptr), d_input(nullptr), d_output(nullptr) {}

    DenseLayer(int in_size, int out_size) {
        init(in_size, out_size);
    }

    void init(int in_size, int out_size) {
        input_size = in_size;
        output_size = out_size;
        h_W.assign(out_size * in_size, 0.0);
        h_b.assign(out_size, 0.0);

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

    ~DenseLayer() {
        if (d_W) cudaFree(d_W);
        if (d_b) cudaFree(d_b);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    // forward: copia input->device, ejecuta kernel y recupera output host
    vector<double> forward(const vector<double>& input) {
        cudaMemcpy(d_input, input.data(), sizeof(double) * input_size, cudaMemcpyHostToDevice);
        forward_kernel<<<output_size, BLOCK_SIZE>>>(d_W, d_b, d_input, d_output, input_size, output_size);
        cudaDeviceSynchronize();
        vector<double> out(output_size);
        cudaMemcpy(out.data(), d_output, sizeof(double) * output_size, cudaMemcpyDeviceToHost);
        return out;
    }

    // backward: recibe grad_out (host) que YA INCLUYE derivada de la activación de la capa
    // y el input que produjo la activación (host). Actualiza W y b, y devuelve grad_input (host).
    vector<double> backward(const vector<double>& input, const vector<double>& grad_out, double lr) {
        double *d_grad_out, *d_grad_in;
        cudaMalloc(&d_grad_out, sizeof(double) * output_size);
        cudaMalloc(&d_grad_in, sizeof(double) * input_size);

        cudaMemcpy(d_grad_out, grad_out.data(), sizeof(double) * output_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input, input.data(), sizeof(double) * input_size, cudaMemcpyHostToDevice);

        // calcular grad_input = W^T * grad_out
        matvec_transpose_kernel<<<input_size, BLOCK_SIZE>>>(d_W, d_grad_out, d_grad_in, input_size, output_size);
        // actualizar pesos
        update_weights_kernel<<<output_size, BLOCK_SIZE>>>(d_W, d_b, d_grad_out, d_input, input_size, output_size, lr);

        cudaDeviceSynchronize();

        vector<double> grad_in(input_size);
        cudaMemcpy(grad_in.data(), d_grad_in, sizeof(double) * input_size, cudaMemcpyDeviceToHost);

        cudaFree(d_grad_out);
        cudaFree(d_grad_in);
        return grad_in;
    }
};

struct SimpleMLP3 {
    DenseLayer l1, l2, out;
    double lr;

    SimpleMLP3(int input_size, int h1, int h2, int output_size, double lr_) : lr(lr_) {
        l1.init(input_size, h1);
        l2.init(h1, h2);
        out.init(h2, output_size);
    }

    vector<double> forward(const vector<double>& x, vector<double>& a1_out, vector<double>& a2_out) {
        a1_out = l1.forward(x);
        a2_out = l2.forward(a1_out);
        return out.forward(a2_out);
    }

    // backward secuencial, recibe x, a1, a2 y target/pred en host
    void backward(const vector<double>& x,
                  const vector<double>& a1,
                  const vector<double>& a2,
                  const vector<double>& pred,
                  const vector<double>& target) {
        int out_sz = (int)pred.size();

        // grad salida: (target - pred) * sigmoid'(pred_input)
        vector<double> grad_out(out_sz);
        for (int i = 0; i < out_sz; ++i) {
            double deriv = pred[i] * (1.0 - pred[i]);
            grad_out[i] = (target[i] - pred[i]) * deriv;
        }

        // Backprop salida -> segunda capa oculta (a2)
        vector<double> grad_a2 = out.backward(a2, grad_out, lr); // grad w.r.t. a2 (antes de derivada)
        // aplicar derivada de activación de a2
        vector<double> grad_out_a2((int)grad_a2.size());
        for (int i = 0; i < (int)grad_a2.size(); ++i)
            grad_out_a2[i] = grad_a2[i] * (a2[i] * (1.0 - a2[i]));

        // Backprop l2 -> a1
        vector<double> grad_a1 = l2.backward(a1, grad_out_a2, lr);
        // aplicar derivada de activación de a1
        vector<double> grad_out_a1((int)grad_a1.size());
        for (int i = 0; i < (int)grad_a1.size(); ++i)
            grad_out_a1[i] = grad_a1[i] * (a1[i] * (1.0 - a1[i]));

        // Backprop l1
        l1.backward(x, grad_out_a1, lr);
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
        vector<double> a1, a2;
        auto outv = forward(x, a1, a2);
        return int(max_element(outv.begin(), outv.end()) - outv.begin());
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

        ofstream log_file("loss_history_mlp3.txt");
        log_file << "epoch,train_loss,val_loss\n";

        int num_batches = (n + batch_size - 1) / batch_size;

        for (int epoch = 0; epoch < epochs; epoch++) {
            random_shuffle(indices.begin(), indices.end());
            double total_loss = 0.0;

            for (int start = 0; start < n; start += batch_size) {
                int end = min(start + batch_size, n);
                double batch_loss = 0.0;

                for (int idx_pos = start; idx_pos < end; idx_pos++) {
                    int idx = indices[idx_pos];
                    vector<double> a1, a2;
                    auto pred = forward(X[idx], a1, a2);

                    vector<double> target(pred.size(), 0.0);
                    target[y[idx]] = 1.0;

                    batch_loss += compute_loss(pred, y[idx]);
                    backward(X[idx], a1, a2, pred, target);
                }
                total_loss += batch_loss / (end - start);
                cudaDeviceSynchronize();
            }

            double avg_train_loss = total_loss / num_batches;
            double val_loss = 0.0;

            if (!X_val.empty()) {
                for (int i = 0; i < (int)X_val.size(); i++) {
                    vector<double> a1, a2;
                    auto pred_val = forward(X_val[i], a1, a2);
                    val_loss += compute_loss(pred_val, y_val[i]);
                }
                val_loss /= X_val.size();
            }

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

// Leer CIFAR-10 batch (compatible con tu función original)
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
    int h1 = 256;
    int h2 = 128;
    int output_size = 10;
    int epochs = 30;
    int minibatch_size = 250;
    double learning_rate = 0.01;

    SimpleMLP3 model(input_size, h1, h2, output_size, learning_rate);

    model.fit(train_images, train_labels, epochs, minibatch_size, test_images, test_labels);

    double train_acc = model.evaluate(train_images, train_labels);
    double test_acc = model.evaluate(test_images, test_labels);

    cout << "Train acc: " << train_acc << "%" << endl;
    cout << "Test acc: " << test_acc << "%" << endl;

    return 0;
}