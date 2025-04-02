#include <torch/torch.h>

struct LSTMDQN : torch::nn::Module {
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    LSTMDQN(int input_dim, int hidden_dim, int output_dim)
        : lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(input_dim, hidden_dim).batch_first(true)))),
          fc1(register_module("fc1", torch::nn::Linear(hidden_dim, 128))),
          fc2(register_module("fc2", torch::nn::Linear(128, output_dim))) {}

    torch::Tensor forward(torch::Tensor x) {
        auto lstm_out = lstm->forward(x);
        auto hidden_state = std::get<0>(lstm_out);
        auto x1 = torch::relu(fc1(hidden_state));
        return fc2(x1);
    }
};

struct ReplayBuffer {
  std::deque<std::tuple<torch::Tensor, int, float, torch::Tensor, bool>> buffer;
  size_t capacity;

  ReplayBuffer(size_t capacity) : capacity(capacity) {}

  void push(torch::Tensor state, int action, float reward, torch::Tensor next_state, bool done) {
      buffer.push_back(std::make_tuple(state, action, reward, next_state, done));
      if (buffer.size() > capacity) {
          buffer.pop_front(); // Remove oldest element when capacity is reached
      }
  }

  std::vector<std::tuple<torch::Tensor, int, float, torch::Tensor, bool>> sample(size_t batch_size) {
      std::vector<std::tuple<torch::Tensor, int, float, torch::Tensor, bool>> batch;
      size_t sample_size = std::min(batch_size, buffer.size());
      
      if (sample_size == 0) return batch;
      
      // Create a temporary copy for sampling
      std::vector<std::tuple<torch::Tensor, int, float, torch::Tensor, bool>> temp(buffer.begin(), buffer.end());
      
      // Shuffle and take the first sample_size elements
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(temp.begin(), temp.end(), g);
      
      return std::vector<std::tuple<torch::Tensor, int, float, torch::Tensor, bool>>(
          temp.begin(), temp.begin() + sample_size);
  }
  
  size_t size() const {
      return buffer.size();
  }
};


void train(LSTMDQN& model, ReplayBuffer& buffer, torch::optim::Adam& optimizer, float gamma, int batch_size) {
  if (buffer.buffer.size() < batch_size) return;  // Wait until buffer is filled

  auto batch = buffer.sample(batch_size);
  std::vector<torch::Tensor> states, next_states;
  std::vector<int> actions;
  std::vector<float> rewards;
  std::vector<bool> dones;

  for (const auto& [s, a, r, ns, d] : batch) {
      states.push_back(s);
      actions.push_back(a);
      rewards.push_back(r);
      next_states.push_back(ns);
      dones.push_back(d);
  }

  // Stack states and next states into tensors
  torch::Tensor state_tensor = torch::stack(states);
  torch::Tensor next_state_tensor = torch::stack(next_states);

  // Prepare action tensor and reward tensor
  torch::Tensor action_tensor = torch::tensor(actions, torch::kLong);
  torch::Tensor reward_tensor = torch::tensor(rewards, torch::kFloat);

  // Convert done flags to torch tensor
  std::vector<uint8_t> dones_uint8(dones.begin(), dones.end());
  torch::Tensor done_tensor = torch::tensor(dones_uint8, torch::kBool);

  // Compute Q values from model
  torch::Tensor q_values = model.forward(state_tensor);
  torch::Tensor next_q_values = model.forward(next_state_tensor).detach();

  // Calculate target Q values: reward + gamma * max(next_q_values)
  torch::Tensor q_target = reward_tensor + gamma * std::get<0>(next_q_values.max(1)) * (1 - done_tensor);

  // Get predicted Q values for taken actions
  torch::Tensor q_pred = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1);

  // Calculate loss
  torch::Tensor loss = torch::mse_loss(q_pred, q_target);

  // Backpropagation
  optimizer.zero_grad();
  loss.backward();
  optimizer.step();
}