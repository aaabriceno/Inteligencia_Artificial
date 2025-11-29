#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <unordered_set>
using namespace std;

#define BLOCK_SIZE 256

vector<double> dropout(const vector<double>& x, double drop_prob) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);

    vector<double> out(x.size());

    for (size_t i = 0; i < x.size(); i++) {
        double r = dist(gen);
        if (r < drop_prob)
            out[i] = 0.0;  // apagado
        else
            out[i] = x[i] / (1.0 - drop_prob);  // escala para mantener E[x]
    }

    return out;
}


vector<double> rotate90(const vector<double>& img) {
    vector<double> out(3072);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int nx = 31 - y;
                int ny = x;
                out[c*1024 + ny*32 + nx] = img[c*1024 + y*32 + x];
            }
        }
    }
    return out;
}

vector<double> flip_horizontal(const vector<double>& img) {
    vector<double> out(3072);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int nx = 31 - x;
                out[c*1024 + y*32 + nx] = img[c*1024 + y*32 + x];
            }
        }
    }
    return out;
}

vector<double> brightness_jitter(const vector<double>& img) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.8, 1.2);

    double factor = dist(gen);
    vector<double> out(3072);

    for (int i = 0; i < 3072; i++) {
        double v = img[i] * factor;
        out[i] = max(-1.0, min(1.0, v));
    }
    return out;
}

vector<double> add_noise(const vector<double>& img) {
    static random_device rd;
    static mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 0.05);

    vector<double> out(3072);
    for (int i = 0; i < 3072; i++) {
        out[i] = img[i] + dist(gen);
        out[i] = max(-1.0, min(1.0, out[i]));
    }
    return out;
}

vector<double> zoom_in(const vector<double>& img) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> crop_dist(24, 30); // tamaño del recorte

    int crop = crop_dist(gen);
    int offset = 32 - crop;

    uniform_int_distribution<int> off_dist(0, offset);
    int ox = off_dist(gen);
    int oy = off_dist(gen);

    // recorte
    vector<double> cropped(crop * crop * 3);

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < crop; y++) {
            for (int x = 0; x < crop; x++) {
                int src_x = x + ox;
                int src_y = y + oy;
                cropped[c*crop*crop + y*crop + x] =
                    img[c*1024 + src_y*32 + src_x];
            }
        }
    }

    // reescalar a 32x32 (nearest neighbor)
    vector<double> out(3072);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int src_x = x * crop / 32;
                int src_y = y * crop / 32;
                out[c*1024 + y*32 + x] =
                    cropped[c * crop * crop + src_y * crop + src_x];
            }
        }
    }

    return out;
}

vector<double> shift_image(const vector<double>& img) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dist(-3, 3);

    int dx = dist(gen);
    int dy = dist(gen);

    vector<double> out(3072, 0.0);

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 32; y++) {
            int ny = y + dy;
            if (ny < 0 || ny >= 32) continue;

            for (int x = 0; x < 32; x++) {
                int nx = x + dx;
                if (nx < 0 || nx >= 32) continue;

                out[c*1024 + ny*32 + nx] =
                    img[c*1024 + y*32 + x];
            }
        }
    }

    return out;
}

vector<double> augment(const vector<double>& img) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, 5);

    int op = dist(gen);

    switch (op) {
        case 0: return flip_horizontal(img);
        case 1: return rotate90(img);
        case 2: return brightness_jitter(img);
        case 3: return add_noise(img);
        case 4: return zoom_in(img);
        case 5: return shift_image(img);
    }
    return img;
}

__host__ __device__ inline double relu(double x) {
    return x > 0 ? x : 0;
}

__host__ __device__ inline double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

__global__ void softmax_kernel(const double* logits, double* soft, int n) {

    int tid = threadIdx.x;

    // 1. Compute max
    __shared__ double maxv;
    if (tid == 0) {
        maxv = logits[0];
        for (int i = 1; i < n; i++)
            maxv = fmax(maxv, logits[i]);
    }
    __syncthreads();

    // 2. Compute exp(x - max)
    __shared__ double sum;
    if (tid == 0) sum = 0.0;
    __syncthreads();

    double val = 0;
    if (tid < n) {
        val = exp(logits[tid] - maxv);
        atomicAdd(&sum, val);
    }
    __syncthreads();

    // 3. Normalize
    if (tid < n) {
        soft[tid] = val / sum;
    }
}

__global__ void ce_grad_kernel(const double* soft, double* dZ, int label, int n) {
    int i = threadIdx.x;
    if (i < n) dZ[i] = soft[i] - (i == label);
}


__global__ void forward_kernel(const double* __restrict__ W,
    const double* __restrict__ b, const double* __restrict__ input,
    double* __restrict__ output, int input_size, int output_size,
    bool apply_relu) {

    int i = blockIdx.x;
    int tid = threadIdx.x;

    if (i >= output_size) return;

    __shared__ double sum[BLOCK_SIZE];
    sum[tid] = 0.0;

    for (int j = tid; j < input_size; j += blockDim.x) {
        sum[tid] += W[i * input_size + j] * input[j];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum[tid] += sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double total = b[i] + sum[0];
        output[i] = apply_relu ? relu(total) : total;
    }
}



__global__ void get_grad_input_kernel(const double* __restrict__ W,
    const double* __restrict__ grad_out, double* __restrict__ grad_in,
    int input_size, int output_size) {

    int j = blockIdx.x;
    int tid = threadIdx.x;
    if (j >= input_size) return;

    __shared__ double sum[BLOCK_SIZE];
    sum[tid] = 0.0;

    for (int i = tid; i < output_size; i += blockDim.x) {
        sum[tid] += W[i * input_size + j] * grad_out[i];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum[tid] += sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) grad_in[j] = sum[0];
}

__global__ void update_weights_kernel(double* __restrict__ W, double* __restrict__ b,
    const double* __restrict__ grad_out, const double* __restrict__ input,
    int input_size, int output_size, double lr) {

    int i = blockIdx.x;
    int tid = threadIdx.x;

    if (i >= output_size) return;

    for (int col = tid; col < input_size; col += blockDim.x) {
        W[i * input_size + col] -= lr * grad_out[i] * input[col];
    }

    if (tid == 0) {
        b[i] -= lr * grad_out[i];
    }
}

__global__ void relu_backward_kernel(const double* grad_out,
    const double* activ, double* grad_in, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    grad_in[i] = grad_out[i] * relu_derivative(activ[i]);
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
        normal_distribution<double> dist(0.0, sqrt(2.0 / input_size));
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

    vector<double> forward(const vector<double>& input, bool apply_relu) {
        cudaMemcpy(d_input, input.data(), sizeof(double) * input_size, cudaMemcpyHostToDevice);

        forward_kernel<<<output_size, BLOCK_SIZE>>>(d_W, d_b, d_input, d_output,
            input_size, output_size, apply_relu);

        cudaDeviceSynchronize();
        vector<double> out(output_size);
        cudaMemcpy(out.data(), d_output, sizeof(double) * output_size, cudaMemcpyDeviceToHost);
        return out;
    }


    vector<double> backward(const vector<double>& input, const vector<double>& grad_out, double lr) {
        double *d_grad_out, *d_grad_in;
        cudaMalloc(&d_grad_out, sizeof(double) * output_size);
        cudaMalloc(&d_grad_in, sizeof(double) * input_size);

        cudaMemcpy(d_grad_out, grad_out.data(), sizeof(double) * output_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input, input.data(), sizeof(double) * input_size, cudaMemcpyHostToDevice);


        get_grad_input_kernel<<<input_size, BLOCK_SIZE>>>(d_W, d_grad_out, d_grad_in, input_size, output_size);

        update_weights_kernel<<<output_size, BLOCK_SIZE>>>(d_W, d_b, d_grad_out, d_input, input_size, output_size, lr);

        cudaDeviceSynchronize();

        vector<double> grad_in(input_size);
        cudaMemcpy(grad_in.data(), d_grad_in, sizeof(double) * input_size, cudaMemcpyDeviceToHost);

        cudaFree(d_grad_out);
        cudaFree(d_grad_in);
        return grad_in;
    }
};

struct MLP {
    DenseLayer l1, l2, out;
    double lr;
    float prob_drop = 0.3;
    MLP(int input_size, int h1, int h2, int output_size, double lr_) : lr(lr_) {
        l1.init(input_size, h1);
        l2.init(h1, h2);
        out.init(h2, output_size);
    }

    vector<double> apply_relu_backward_cuda(const vector<double>& grad_out,
        const vector<double>& activ){

        int n = grad_out.size();

        double *d_grad_out, *d_activ, *d_grad_in;
        cudaMalloc(&d_grad_out, sizeof(double)*n);
        cudaMalloc(&d_activ,    sizeof(double)*n);
        cudaMalloc(&d_grad_in,  sizeof(double)*n);

        cudaMemcpy(d_grad_out, grad_out.data(), sizeof(double)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_activ,    activ.data(),    sizeof(double)*n, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        relu_backward_kernel<<<blocks, threads>>>(d_grad_out, d_activ, d_grad_in, n);

        vector<double> grad_in(n);
        cudaMemcpy(grad_in.data(), d_grad_in, sizeof(double)*n, cudaMemcpyDeviceToHost);

        cudaFree(d_grad_out);
        cudaFree(d_activ);
        cudaFree(d_grad_in);

        return grad_in;
    }

    vector<double> forward(const vector<double>& x, vector<double>& a1_out, vector<double>& a2_out, bool training) {

        a1_out = l1.forward(x, true);
        if (training)
            a1_out = dropout(a1_out, prob_drop);

        a2_out = l2.forward(a1_out, true);
        if (training)
            a2_out = dropout(a2_out, prob_drop);

        vector<double> logits = out.forward(a2_out, false);

        int C = logits.size();
        double *d_logits, *d_soft;

        cudaMalloc(&d_logits, sizeof(double)*C);
        cudaMalloc(&d_soft,   sizeof(double)*C);
        cudaMemcpy(d_logits, logits.data(), sizeof(double)*C, cudaMemcpyHostToDevice);

        softmax_kernel<<<1, C>>>(d_logits, d_soft, C);

        vector<double> soft(C);
        cudaMemcpy(soft.data(), d_soft, sizeof(double)*C, cudaMemcpyDeviceToHost);

        cudaFree(d_logits);
        cudaFree(d_soft);

        return soft;
    }


    vector<double> backward(const vector<double>& x,
                    const vector<double>& a1,
                    const vector<double>& a2,
                    const vector<double>& soft,
                    const vector<double>& target) {

        int C = soft.size();
        double *d_soft, *d_dZ;
        cudaMalloc(&d_soft, sizeof(double)*C);
        cudaMalloc(&d_dZ,   sizeof(double)*C);

        cudaMemcpy(d_soft, soft.data(), sizeof(double)*C, cudaMemcpyHostToDevice);

        int label = -1;
        for (int i = 0; i < C; i++)
            if (target[i] == 1.0) label = i;

        ce_grad_kernel<<<1, C>>>(d_soft, d_dZ, label, C);

        vector<double> grad_out(C);
        cudaMemcpy(grad_out.data(), d_dZ, sizeof(double)*C, cudaMemcpyDeviceToHost);

        cudaFree(d_soft);
        cudaFree(d_dZ);

        vector<double> grad_a2 = out.backward(a2, grad_out, lr);

        vector<double> grad_out_a2 = apply_relu_backward_cuda(grad_a2, a2);
        vector<double> grad_a1 = l2.backward(a1, grad_out_a2, lr);

        vector<double> grad_out_a1 = apply_relu_backward_cuda(grad_a1, a1);
        return l1.backward(x, grad_out_a1, lr);
    }

    double compute_loss(const vector<double>& pred, int label) {
        return -log(max(pred[label], 1e-12));
    }


    int predict(const vector<double>& x) {
        vector<double> a1, a2;
        auto outv = forward(x, a1, a2, false);
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

        ofstream log_file("loss_history.txt");
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
                    vector<double> xin = augment(X[idx]);
                    auto pred = forward(xin, a1, a2, true);

                    vector<double> target(pred.size(), 0.0);
                    target[y[idx]] = 1.0;

                    batch_loss += compute_loss(pred, y[idx]);
                    backward(xin, a1, a2, pred, target);
                }
                total_loss += batch_loss / (end - start);
                cudaDeviceSynchronize();
            }

            double avg_train_loss = total_loss / num_batches;
            double val_loss = 0.0;

            if (!X_val.empty()) {
                for (int i = 0; i < (int)X_val.size(); i++) {
                    vector<double> a1, a2;
                    auto pred_val = forward(X_val[i], a1, a2,false);
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
        string fname = "cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin";
        load_cifar10_batch(fname, train_images, train_labels);
    }
    load_cifar10_batch("cifar-10-batches-bin/test_batch.bin", test_images, test_labels);

    cout << "Cargadas " << train_images.size() << " train y " << test_images.size() << " test" << endl;

    int input_size = 32 * 32 * 3;
    int h1 = 512;
    int h2 = 128;
    int output_size = 10;
    int epochs = 500;
    int minibatch_size = 250;
    double learning_rate = 0.0001;

    MLP model(input_size, h1, h2, output_size, learning_rate);

    model.fit(train_images, train_labels, epochs, minibatch_size, test_images, test_labels);
    //model.fit(train_images, train_labels, epochs, minibatch_size);

    double train_acc = model.evaluate(train_images, train_labels);
    double test_acc = model.evaluate(test_images, test_labels);

    cout << "Train acc: " << train_acc << "%" << endl;
    cout << "Test acc: " << test_acc << "%" << endl;

    return 0;

}
