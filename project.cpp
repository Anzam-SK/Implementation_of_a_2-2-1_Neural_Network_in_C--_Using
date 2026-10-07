#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <utility>
#include <string>

using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>; // rows x cols

// ---------- Math helpers ----------
static inline double sigmoid_scalar(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

static Vec sigmoid_vec(const Vec& z) {
    Vec a(z.size());
    for (size_t i = 0; i < z.size(); ++i) a[i] = sigmoid_scalar(z[i]);
    return a;
}

static Vec sigmoid_derivative_from_z(const Vec& z) {
    Vec s(z.size());
    for (size_t i = 0; i < z.size(); ++i) {
        double a = sigmoid_scalar(z[i]);
        s[i] = a * (1.0 - a);
    }
    return s;
}

static Vec mat_vec(const Mat& W, const Vec& v) {
    // W: (rows x cols), v: (cols)
    size_t rows = W.size();
    size_t cols = W.empty() ? 0 : W[0].size();
    Vec out(rows, 0.0);
    for (size_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (size_t c = 0; c < cols; ++c) acc += W[r][c] * v[c];
        out[r] = acc;
    }
    return out;
}

static Vec matT_vec(const Mat& W, const Vec& v) {
    // W^T * v  (W: rows x cols, v: rows)
    size_t rows = W.size();
    size_t cols = W.empty() ? 0 : W[0].size();
    Vec out(cols, 0.0);
    for (size_t c = 0; c < cols; ++c) {
        double acc = 0.0;
        for (size_t r = 0; r < rows; ++r) acc += W[r][c] * v[r];
        out[c] = acc;
    }
    return out;
}

static Mat outer(const Vec& a, const Vec& b) {
    // a: m, b: n  => m x n
    Mat out(a.size(), Vec(b.size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            out[i][j] = a[i] * b[j];
    return out;
}

static void axpy_inplace(Mat& W, const Mat& G, double alpha) {
    // W = W + alpha * G
    for (size_t i = 0; i < W.size(); ++i)
        for (size_t j = 0; j < W[i].size(); ++j)
            W[i][j] += alpha * G[i][j];
}

static Mat zeros_like(const Mat& W) {
    Mat Z = W;
    for (auto& row : Z) for (auto& x : row) x = 0.0;
    return Z;
}

// ---------- Data ----------
static void create_xor_dataset(vector<Vec>& input_data, vector<Vec>& target_data) {
    input_data = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    target_data = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
}

static vector<Mat> initialize_network_weights(const vector<int>& layer_sizes, double scale, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    vector<Mat> weights;
    for (size_t i = 0; i + 1 < layer_sizes.size(); ++i) {
        int rows = layer_sizes[i + 1];
        int cols = layer_sizes[i];
        Mat W(rows, Vec(cols, 0.0));
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                W[r][c] = scale * (dist(rng) - 0.5);
        weights.push_back(std::move(W));
    }
    return weights;
}

// ---------- Forward / Backward ----------
static pair<vector<Vec>, vector<Vec>> forward_pass(const vector<Mat>& weights, const Vec& input_col) {
    vector<Vec> z_values;
    vector<Vec> activations;
    activations.push_back(input_col); // a^0

    for (const auto& W : weights) {
        Vec z = mat_vec(W, activations.back());
        Vec a = sigmoid_vec(z);
        z_values.push_back(std::move(z));
        activations.push_back(std::move(a));
    }
    return { z_values, activations };
}

static vector<Vec> backward_pass(const vector<Mat>& weights,
    const vector<Vec>& z_values,
    const vector<Vec>& activations,
    const Vec& target_output) {
    size_t L = weights.size();
    vector<Vec> deltas(L);

    // Output layer delta: (a^L - y) * sigmoid'(z^L)
    Vec d_out(activations.back().size());
    Vec sigp_L = sigmoid_derivative_from_z(z_values.back());
    for (size_t i = 0; i < d_out.size(); ++i) {
        d_out[i] = (activations.back()[i] - target_output[i]) * sigp_L[i];
    }
    deltas[L - 1] = std::move(d_out);

    // Hidden layers: sigmoid'(z^l) * (W^{l+1}^T * delta^{l+1})
    for (int l = static_cast<int>(L) - 2; l >= 0; --l) {
        Vec sigp = sigmoid_derivative_from_z(z_values[l]);
        Vec back = matT_vec(weights[l + 1], deltas[l + 1]);
        Vec d(sigp.size());
        for (size_t i = 0; i < d.size(); ++i) d[i] = sigp[i] * back[i];
        deltas[l] = std::move(d);
    }
    return deltas;
}

// ---------- Updates ----------
static void apply_weight_updates(vector<Mat>& weights,
    const vector<Vec>& deltas,
    const vector<Vec>& activations,
    double learning_rate) {
    for (size_t l = 0; l < weights.size(); ++l) {
        Mat grad = outer(deltas[l], activations[l]); // d_l * a_{l}^T
        axpy_inplace(weights[l], grad, -learning_rate);
    }
}

// ---------- Loss / Predict ----------
static double mse_epoch(const vector<Mat>& weights,
    const vector<Vec>& input_data,
    const vector<Vec>& target_data) {
    double total = 0.0;
    for (size_t i = 0; i < input_data.size(); ++i) {
        auto fa = forward_pass(weights, input_data[i]);
        const Vec& pred = fa.second.back();
        const Vec& tgt = target_data[i];
        double se = 0.0;
        for (size_t k = 0; k < pred.size(); ++k) {
            double diff = pred[k] - tgt[k];
            se += 0.5 * diff * diff;
        }
        total += se;
    }
    return total / static_cast<double>(input_data.size());
}

static Vec predict_one(const vector<Mat>& weights, const Vec& x) {
    auto fa = forward_pass(weights, x);
    return fa.second.back();
}

// ---------- Training Loops ----------
static pair<vector<Mat>, vector<double>> stochastic_gradient_descent(vector<Mat> weights,
    const vector<Vec>& input_data,
    const vector<Vec>& target_data,
    double learning_rate = 0.5,
    int num_epochs = 20000,
    int log_interval = 2000) {
    vector<double> epoch_losses;
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        // Online update per sample
        for (size_t i = 0; i < input_data.size(); ++i) {
            auto fa = forward_pass(weights, input_data[i]);
            vector<Vec> deltas = backward_pass(weights, fa.first, fa.second, target_data[i]);
            apply_weight_updates(weights, deltas, fa.second, learning_rate);
        }

        double loss = mse_epoch(weights, input_data, target_data);
        epoch_losses.push_back(loss);
        if (epoch % log_interval == 0) {
            cout << "[Online] Epoch " << setw(5) << epoch << " | MSE: "
                << fixed << setprecision(6) << loss << "\n";
        }
    }
    return { weights, epoch_losses };
}

static pair<vector<Mat>, vector<double>> batch_gradient_descent(vector<Mat> weights,
    const vector<Vec>& input_data,
    const vector<Vec>& target_data,
    double learning_rate = 0.5,
    int num_epochs = 10000,
    int log_interval = 1000) {
    vector<double> epoch_losses;
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        // Accumulate gradients
        vector<Mat> grad_acc;
        grad_acc.reserve(weights.size());
        for (const auto& W : weights) grad_acc.push_back(zeros_like(W));

        double total_se = 0.0;
        for (size_t i = 0; i < input_data.size(); ++i) {
            auto fa = forward_pass(weights, input_data[i]);
            vector<Vec> deltas = backward_pass(weights, fa.first, fa.second, target_data[i]);
            // accumulate grad = delta * a_prev^T
            for (size_t l = 0; l < weights.size(); ++l) {
                Mat g = outer(deltas[l], fa.second[l]);
                axpy_inplace(grad_acc[l], g, 1.0);
            }
            const Vec& pred = fa.second.back();
            const Vec& tgt = target_data[i];
            double se = 0.0;
            for (size_t k = 0; k < pred.size(); ++k) {
                double diff = pred[k] - tgt[k];
                se += 0.5 * diff * diff;
            }
            total_se += se;
        }

        // Apply averaged gradients
        double invN = 1.0 / static_cast<double>(input_data.size());
        for (size_t l = 0; l < weights.size(); ++l) {
            axpy_inplace(weights[l], grad_acc[l], -learning_rate * invN);
        }

        double loss = total_se * invN;
        epoch_losses.push_back(loss);
        if (epoch % log_interval == 0) {
            cout << "[Batch ] Epoch " << setw(5) << epoch << " | MSE: "
                << fixed << setprecision(6) << loss << "\n";
        }
    }
    return { weights, epoch_losses };
}

// ---------- Pretty printing ----------
static void print_input_vec_like_numpy(const Vec& v) {
    cout << "input=[";
    cout << fixed << setprecision(1);
    for (size_t i = 0; i < v.size(); ++i) {
        cout << v[i];
        if (i + 1 < v.size()) cout << " ";
    }
    cout.unsetf(std::ios::floatfield);
    cout << "] ";
}

static double calc_mse_final(const vector<Mat>& weights,
    const vector<Vec>& input_data,
    const vector<Vec>& target_data) {
    return mse_epoch(weights, input_data, target_data);
}

// ---------- Main / Experiment ----------
int main() {
    // Seed to mirror structure (not identical NumPy RNG but deterministic)
    std::mt19937 rng(42);

    vector<Vec> input_data, target_data;
    create_xor_dataset(input_data, target_data);

    // Architecture: 2 -> 2 -> 1
    vector<int> layer_sizes = { 2, 2, 1 };

    // Two independent weight sets
    vector<Mat> weights_online = initialize_network_weights(layer_sizes, 1.0, rng);
    vector<Mat> weights_batch = initialize_network_weights(layer_sizes, 1.0, rng);

    cout << "=== Stochastic (Online) Training ===\n";
    auto online = stochastic_gradient_descent(weights_online, input_data, target_data,
        /*learning_rate=*/0.5, /*num_epochs=*/8000, /*log_interval=*/2000);
    weights_online = std::move(online.first);

    cout << "\n=== Batch Training ===\n";
    auto batch = batch_gradient_descent(weights_batch, input_data, target_data,
        /*learning_rate=*/0.7, /*num_epochs=*/5000, /*log_interval=*/1000);
    weights_batch = std::move(batch.first);

    // Final Predictions (Online)
    cout << "\nFinal Predictions (Stochastic):\n";
    for (size_t i = 0; i < input_data.size(); ++i) {
        Vec p = predict_one(weights_online, input_data[i]);
        print_input_vec_like_numpy(input_data[i]);
        cout << "-> prediction=" << fixed << setprecision(4) << p[0]
            << " -> class=" << ((p[0] > 0.5) ? 1 : 0) << "\n";
    }

    // Final Predictions (Batch)
    cout << "\nFinal Predictions (Batch):\n";
    for (size_t i = 0; i < input_data.size(); ++i) {
        Vec p = predict_one(weights_batch, input_data[i]);
        print_input_vec_like_numpy(input_data[i]);
        cout << "-> prediction=" << fixed << setprecision(4) << p[0]
            << " -> class=" << ((p[0] > 0.5) ? 1 : 0) << "\n";
    }

    // Final MSEs
    double mse_online = calc_mse_final(weights_online, input_data, target_data);
    double mse_batch = calc_mse_final(weights_batch, input_data, target_data);

    cout << "\nFinal MSE (Stochastic): " << fixed << setprecision(6) << mse_online << "\n";
    cout << "Final MSE (Batch):      " << fixed << setprecision(6) << mse_batch << "\n";

    return 0;
}
