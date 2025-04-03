#include <iostream>
#include <torch/torch.h>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include "model.cpp"
#include "environment.cpp"

using namespace std;
using namespace torch;
using namespace torch::nn;

int select_action(LSTMDQN& model, Tensor state, float epsilon) {
  random_device rd;
  std::mt19937 gen(rd());
  uniform_real_distribution<float> dist(0.0f, 1.0f);
  
  if (dist(gen) < epsilon) {
      uniform_int_distribution<int> action_dist(0, 3);
      return action_dist(gen);
  } else {
      NoGradGuard no_grad;
      Tensor q_values = model.forward(state);
      return q_values.argmax(1).item<int>();
  }
}

void evaluate(LSTMDQN& model, Environment& env, int num_episodes = 5) {
  float total_reward = 0.0f;
  
  for (int episode = 0; episode < num_episodes; episode++) {
      Tensor state = env.reset();
      bool done = false;
      float episode_reward = 0.0f;
      
      while (!done) {
          NoGradGuard no_grad;
          int action = select_action(model, state, 0.0f);
          
          UpdateData update = env.forward(action);
          episode_reward += update.reward;
          
          done = update.done;
          if (done) break;
          
          state = update.state;
      }
      
      total_reward += episode_reward;
      cout << "Evaluation Episode " << episode << " Reward: " << episode_reward << endl;
  }
  
  cout << "Average Evaluation Reward: " << total_reward / num_episodes << endl;
}


int main(int argc, char* argv[]){
  try {
    string datapath = argv[1];
    Environment env(datapath, 30);
    int state_dim = env.get_state_dim();
    int action_dim = 4;
    int hidden_dim = 64;

    int lr = 0.001;

    cout << "State dimension: " << state_dim << endl;

    LSTMDQN model(state_dim, hidden_dim, action_dim);
    ReplayBuffer replay_buffer(10000);
    optim::Adam optimizer(model.parameters(), optim::AdamOptions(lr));


    float gamma = 0.99;
    int batch_size = 32;
    int num_episodes = 1000;
    
    float epsilon_start = 1.0f;
    float epsilon_end = 0.01f;
    float epsilon_decay = 0.995f;
    float epsilon = epsilon_start;

    cout << "Starting Episode Loop" << endl;
    for (int episode = 0; episode < num_episodes; episode++) {
      Tensor state = env.reset();
      cout << state << endl;
      bool done = false;
      float episode_reward = 0.0f;

      cout << "Starting Epoch Loop" << endl;
      while (!done) {
        cout << "Selecting_Action" << endl;
        int action = select_action(model, state, epsilon);
        cout << "Stepping Environment" << endl;
        UpdateData update = env.forward(action);
        cout << update.state << endl;
        episode_reward += update.reward;
        
        cout << "Pushing_Memory" << endl;
        replay_buffer.push(state, action, update.reward, update.state, update.done);

        cout << "Training" << endl;
        train(model, replay_buffer, optimizer, gamma, batch_size);

        done = update.done;
        if (done) break;
        
        state = update.state;
      }

      epsilon = max(epsilon_end, epsilon * epsilon_decay);
      if (episode % 10 == 0) {
        cout << "Episode " << episode << " Reward: " << episode_reward 
                  << " Epsilon: " << epsilon 
                  << " Buffer Size: " << replay_buffer.size() << endl;
      }
    
      if (episode % 100 == 0) {
        cout << "Evaluating model..." << endl;
        evaluate(model, env);
      }
    }

  } catch (const exception& e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }
  return 0;
}
