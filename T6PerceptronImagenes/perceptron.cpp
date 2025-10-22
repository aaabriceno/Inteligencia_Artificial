#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <string>

using namespace std;

double sigmoide(double x) {
    return 1.0 / (1.0 + exp(-x));
}

struct CapaDensa {
    int input_size, output_size;
    vector<double> W, b;

    CapaDensa(int in_size, int out_size) {
        input_size = in_size;
        output_size = out_size;

        W.resize(out_size * in_size);
        b.resize(out_size, 0.0);

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0, 0.01);

        for (auto &w : W) w = dist(gen);
    }

    vector<double> forward(const vector<double>& input) {
        vector<double> output(output_size, 0.0);
        for (int i = 0; i < output_size; i++) {
            double sum = b[i];
            for (int j = 0; j < input_size; j++) {
                sum += W[i * input_size + j] * input[j];
            }
            output[i] = sigmoide(sum);
        }
        return output;
    }

    void backward(const vector<double>& input, const vector<double>& grad_out, double lr) {
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                double grad = grad_out[i] * input[j];
                W[i * input_size + j] += lr * grad;
            }
            b[i] += lr * grad_out[i];
        }
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
        for (int i = 0; i < pred.size(); i++) {
            double t = (i == label ? 1.0 : 0.0);
            loss += pow(pred[i] - t, 2);
        }
        return loss / pred.size();
    }

    int predict(const vector<double>& x) {
        vector<double> out = forward(x);
        return max_element(out.begin(), out.end()) - out.begin();
    }

    double evaluate(const vector<vector<double>>& X, const vector<int>& y) {
        int correct = 0;
        for (int i = 0; i < X.size(); i++) {
            if (predict(X[i]) == y[i]) correct++;
        }
        return (double)correct / X.size() * 100.0;
    }

    void fit(const vector<vector<double>>& X, const vector<int>& y,
             int epochs, int batch_size,
             const vector<vector<double>>& X_val = {},
             const vector<int>& y_val = {}) {
        int n = X.size();
        vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);

        ofstream log_file("loss_history.txt");
        if (!log_file.is_open()) {
            cerr << "No se pudo abrir loss_log.txt para escribir." << endl;
            return;
        }
        log_file << "epoch,train_loss,val_loss\n";

        for (int epoch = 0; epoch < epochs; epoch++) {
            random_device rd_epoch;
            mt19937 g(rd_epoch());
            shuffle(indices.begin(), indices.end(), g);
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
            }

            double avg_train_loss = total_loss / (n / batch_size);
            double val_loss = 0.0;

            if (!X_val.empty()) {
                for (int i = 0; i < X_val.size(); i++) {
                    auto pred_val = forward(X_val[i]);
                    val_loss += compute_loss(pred_val, y_val[i]);
                }
                val_loss /= X_val.size();
            }

            cout << "Epoch " << epoch + 1
                 << " - Train Loss: " << fixed << setprecision(4) << avg_train_loss;
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


void load_cifar10_batch(const string& filename,
                        vector<vector<double>>& images,
                        vector<int>& labels) {
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
        for (int j = 0; j < image_size; j++) {
            img[j] = (buffer[j] / 255.0 - 0.5) * 2.0; // [-1,1]
        }
        images.push_back(img);
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

    cout << "Cargadas " << train_images.size() << " train y "
         << test_images.size() << " test" << endl;

    int input_size = 32 * 32 * 3;
    int output_size = 10;
    int epochs = 100;
    int minibatch_size = 250;
    double learning_rate = 0.005;

    SimpleMLP model(input_size, output_size, learning_rate);

    model.fit(train_images, train_labels, epochs, minibatch_size, test_images, test_labels);

    double train_acc = model.evaluate(train_images, train_labels);
    double test_acc = model.evaluate(test_images, test_labels);

    cout << "Train acc: " << train_acc << "%" << endl;
    cout << "Test acc: " << test_acc << "%" << endl;

    return 0;
}
