#include <iostream>
#include <torch/script.h>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: use-torch-model <path-to-exported-script-module>\n";
        return -1;
    }


    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    int64_t n = 10;
    torch::manual_seed(0);
    auto X = torch::randn({n, 3});
    auto X_keys = torch::randint(2, {n, 2});

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(X);
    inputs.push_back(X_keys);

    at::Tensor output = module.forward(inputs).toTensor();


    std::cout << "X\n" << X << std::endl;
    std::cout << "X_keys\n" << X_keys << std::endl;
    std::cout << std::endl;
    std::cout << "reduced X\n" << output << std::endl;

    return 0;
}
