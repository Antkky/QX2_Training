#include <iostream>
#include <torch/torch.h>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include "model.cpp"
#include "environment.cpp"

int select_action(LSTMDQN& model, torch::Tensor state, float epsilon) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  
  if (dist(gen) < epsilon) {
      std::uniform_int_distribution<int> action_dist(0, 3);
      return action_dist(gen);
  } else {
      torch::NoGradGuard no_grad;
      torch::Tensor q_values = model.forward(state);
      return q_values.argmax(1).item<int>();
  }
}

void evaluate(LSTMDQN& model, Environment& env, int num_episodes = 5) {
  float total_reward = 0.0f;
  
  for (int episode = 0; episode < num_episodes; episode++) {
      torch::Tensor state = env.reset();
      bool done = false;
      float episode_reward = 0.0f;
      
      while (!done) {
          torch::NoGradGuard no_grad;
          int action = select_action(model, state, 0.0f);
          
          UpdateData update = env.forward(action);
          episode_reward += update.reward;
          
          done = update.done;
          if (done) break;
          
          state = update.state;
      }
      
      total_reward += episode_reward;
      std::cout << "Evaluation Episode " << episode << " Reward: " << episode_reward << std::endl;
  }
  
  std::cout << "Average Evaluation Reward: " << total_reward / num_episodes << std::endl;
}


int main(){
  try {
    Environment env("./data/processed/train_data.csv", 10);
    int state_dim = env.get_state_dim();
    int action_dim = 4;
    int hidden_dim = 64;

    int lr = 0.001;

    std::cout << "State dimension: " << state_dim << std::endl;

    LSTMDQN model(state_dim, hidden_dim, action_dim);
    ReplayBuffer replay_buffer(10000);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(lr));


    float gamma = 0.99;
    int batch_size = 32;
    int num_episodes = 1000;
    
    float epsilon_start = 1.0f;
    float epsilon_end = 0.01f;
    float epsilon_decay = 0.995f;
    float epsilon = epsilon_start;

    for (int episode = 0; episode < num_episodes; episode++) {
      torch::Tensor state = env.reset();
      bool done = false;
      float episode_reward = 0.0f;

      while (!done) {
        int action = select_action(model, state, epsilon);
        UpdateData update = env.forward(action);
        episode_reward += update.reward;
        
        replay_buffer.push(state, action, update.reward, update.state, update.done);

        train(model, replay_buffer, optimizer, gamma, batch_size);

        done = update.done;
        if (done) break;
        
        state = update.state;
      }

      epsilon = std::max(epsilon_end, epsilon * epsilon_decay);
      if (episode % 10 == 0) {
        std::cout << "Episode " << episode << " Reward: " << episode_reward 
                  << " Epsilon: " << epsilon 
                  << " Buffer Size: " << replay_buffer.size() << std::endl;
      }
    
      if (episode % 100 == 0) {
        std::cout << "Evaluating model..." << std::endl;
        evaluate(model, env);
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
