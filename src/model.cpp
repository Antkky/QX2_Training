#include <torch/torch.h>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>

using namespace std;
using namespace torch;
using namespace torch::nn;

struct LSTMDQN : Module {
    LSTM lstm{nullptr};
    Linear fc1{nullptr};
    Linear fc2{nullptr};

    LSTMDQN(int input_dim, int hidden_dim, int output_dim)
        : lstm(register_module("lstm", LSTM(LSTMOptions(input_dim, hidden_dim).batch_first(true)))),
          fc1(register_module("fc1", Linear(hidden_dim, 128))),
          fc2(register_module("fc2", Linear(128, output_dim))) {}

    Tensor forward(Tensor x) {
        auto lstm_out = lstm->forward(x);
        auto hidden_state = get<0>(lstm_out);
        auto x1 = relu(fc1(hidden_state));
        return fc2(x1);
    }
};

struct ReplayBuffer {
  deque<tuple<Tensor, int, float, Tensor, bool>> buffer;
  size_t capacity;

  ReplayBuffer(size_t capacity) : capacity(capacity) {}

  void push(Tensor state, int action, float reward, Tensor next_state, bool done) {
      buffer.push_back(make_tuple(state, action, reward, next_state, done));
      if (buffer.size() > capacity) {
          buffer.pop_front();
      }
  }

  vector<tuple<Tensor, int, float, Tensor, bool>> sample(size_t batch_size) {
      vector<tuple<Tensor, int, float, Tensor, bool>> batch;
      size_t sample_size = min(batch_size, buffer.size());
      
      if (sample_size == 0) return batch;
      
      vector<tuple<Tensor, int, float, Tensor, bool>> temp(buffer.begin(), buffer.end());
      
      random_device rd;
      std::mt19937 g(rd());
      shuffle(temp.begin(), temp.end(), g);
      
      return vector<tuple<Tensor, int, float, Tensor, bool>>(
          temp.begin(), temp.begin() + sample_size);
  }
  
  size_t size() const {
      return buffer.size();
  }
};


void train(LSTMDQN& model, ReplayBuffer& buffer, optim::Adam& optimizer, float gamma, int batch_size) {
  if (buffer.buffer.size() < batch_size) return;

  auto batch = buffer.sample(batch_size);
  vector<Tensor> states, next_states;
  vector<int> actions;
  vector<float> rewards;
  vector<bool> dones;

  for (const auto& [s, a, r, ns, d] : batch) {
      states.push_back(s);
      actions.push_back(a);
      rewards.push_back(r);
      next_states.push_back(ns);
      dones.push_back(d);
  }

  Tensor state_tensor = torch::stack(states);
  Tensor next_state_tensor = torch::stack(next_states);

  Tensor action_tensor = torch::tensor(actions, kLong);
  Tensor reward_tensor = torch::tensor(rewards, kFloat);

  vector<uint8_t> dones_uint8(dones.begin(), dones.end());
  Tensor done_tensor = torch::tensor(dones_uint8, kBool);

  Tensor q_values = model.forward(state_tensor);
  Tensor next_q_values = model.forward(next_state_tensor).detach();

  Tensor q_target = reward_tensor + gamma * get<0>(next_q_values.max(1)) * (1 - done_tensor);

  Tensor q_pred = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1);

  Tensor loss = mse_loss(q_pred, q_target);

  optimizer.zero_grad();
  loss.backward();
  optimizer.step();
}