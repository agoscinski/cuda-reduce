#include <iostream>
#include <torch/torch.h>
#include "reduce.cpp"

int main() {
  int64_t n{10};
  torch::manual_seed(0);
  torch::Tensor X{torch::randn({n, 3})};
  torch::Tensor X_keys{torch::randint(2, {n, 2})};
  int64_t col{1};
  torch::Tensor out{reduce(X, X_keys, col)};
  std::cout << "X\n" << X << std::endl;
  std::cout << "X_keys\n" << X_keys << std::endl;
  std::cout << "col " << col << std::endl;
  std::cout << std::endl;
  std::cout << "reduced X\n" << out << std::endl;
  return 0;
}
