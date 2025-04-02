#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

// struct for each environment forward pass
struct UpdateData
{
  torch::Tensor state;
  float reward;
  bool done;
};

// make sure everything in unscaled, we will scale the necessary parts later-on
struct ImportData {
  std::string time;
  float asset1_price;
  float asset2_price;
  float ratio_price;
  float hedge_ratio;
  float spread_zscore;
  float correlation;
  float cointegration;
  float rsi1;
  float rsi2;
  float rsi3;
  float macd1;
  float macd2;
  float macd3;
};

// dynamic columns for later-on
struct DynamicData {
  int streak;
  int positioning;
  float unrealized;
  float realized;
};

class Environment
{
private:
  std::vector<ImportData> data;
  std::vector<DynamicData> dynamic_data;
  int current_step;
  int window_size;
  int data_size;
  int positioning;  // Current position: -1 (short), 0 (neutral), 1 (long)
  float total_reward;
  float total_profit;
  const int state_dim = 17;  // Define the state dimension explicitly
  
  void execute(int action);
  float calculate_reward();
  torch::Tensor current_state();
  bool parse_csv_line(const std::string& line, ImportData& data);
  bool load_data(const std::string& filepath);

public:
  Environment(std::string filepath, int window_size);
  ~Environment();
  UpdateData forward(int action);
  torch::Tensor reset();
  int get_state_dim() const { return state_dim; }
};

Environment::Environment(std::string filepath, int window_size) : window_size(window_size), current_step(0), positioning(0), total_reward(0.0f), total_profit(0.0f)
{
  // Load data from file
  if (!load_data(filepath)) {
    std::cerr << "Failed to load data from " << filepath << std::endl;
    throw std::runtime_error("Data loading failed");
  }
  
  data_size = data.size();
  
  // Initialize dynamic data
  dynamic_data.resize(data_size);
  for (auto& dd : dynamic_data) {
    dd.streak = 0;
    dd.positioning = 0;
    dd.unrealized = 0.0f;
    dd.realized = 0.0f;
  }
  
  std::cout << "Environment initialized with " << data_size << " data points" << std::endl;
}

Environment::~Environment()
{
  // Cleanup if needed
}

bool Environment::parse_csv_line(const std::string& line, ImportData& data_point) {
  std::stringstream ss(line);
  std::string cell;
  
  // Parse time
  if (!std::getline(ss, cell, ',')) return false;
  data_point.time = cell;
  
  // Parse numeric values
  try {
    if (!std::getline(ss, cell, ',')) return false;
    data_point.asset1_price = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.asset2_price = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.ratio_price = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.hedge_ratio = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.spread_zscore = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.correlation = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.cointegration = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.rsi1 = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.rsi2 = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.rsi3 = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.macd1 = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.macd2 = std::stof(cell);
    
    if (!std::getline(ss, cell, ',')) return false;
    data_point.macd3 = std::stof(cell);
    
  } catch (const std::exception& e) {
    std::cerr << "Error parsing line: " << e.what() << std::endl;
    return false;
  }
  
  return true;
}

bool Environment::load_data(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filepath << std::endl;
    return false;
  }
  
  std::string line;
  // Skip header
  std::getline(file, line);
  
  while (std::getline(file, line)) {
    ImportData data_point;
    if (parse_csv_line(line, data_point)) {
      data.push_back(data_point);
    }
  }
  
  return !data.empty();
}

torch::Tensor Environment::reset()
{
  current_step = window_size; // Start after having enough history for the window
  positioning = 0;
  total_reward = 0.0f;
  total_profit = 0.0f;
  
  // Reset dynamic data
  for (auto& dd : dynamic_data) {
    dd.streak = 0;
    dd.positioning = 0;
    dd.unrealized = 0.0f;
    dd.realized = 0.0f;
  }
  
  return current_state();
}

torch::Tensor Environment::current_state()
{
  // Create state tensor
  // Structure: [market_features (from ImportData) + position_features (from DynamicData)]
  std::vector<float> state_vec;
  
  // Add window of spread z-scores
  for (int i = current_step - window_size; i < current_step; i++) {
    state_vec.push_back(data[i].spread_zscore);
  }
  
  // Add current values
  state_vec.push_back(data[current_step].spread_zscore);
  state_vec.push_back(data[current_step].correlation);
  state_vec.push_back(data[current_step].rsi1);
  state_vec.push_back(data[current_step].rsi2);
  state_vec.push_back(data[current_step].macd1);
  
  // Add position info
  state_vec.push_back(static_cast<float>(positioning));
  state_vec.push_back(dynamic_data[current_step].unrealized);
  
  // Create tensor and reshape for LSTM input [batch_size, seq_length, input_size]
  torch::Tensor state = torch::tensor(state_vec).reshape({1, 1, state_dim});
  
  return state;
}

void Environment::execute(int action)
{
  // 0: No action, 1: Go Long, 2: Go Short, 3: Exit Position
  int prev_position = positioning;
  
  switch (action) {
    case 0: // No action
      break;
    case 1: // Go Long
      if (positioning != 1) {
        // Close short position if exists
        if (positioning == -1) {
          dynamic_data[current_step].realized = dynamic_data[current_step-1].unrealized;
          dynamic_data[current_step].streak = 0;
        }
        positioning = 1;
      }
      break;
    case 2: // Go Short
      if (positioning != -1) {
        // Close long position if exists
        if (positioning == 1) {
          dynamic_data[current_step].realized = dynamic_data[current_step-1].unrealized;
          dynamic_data[current_step].streak = 0;
        }
        positioning = -1;
      }
      break;
    case 3: // Exit Position
      if (positioning != 0) {
        dynamic_data[current_step].realized = dynamic_data[current_step-1].unrealized;
        positioning = 0;
        dynamic_data[current_step].streak = 0;
      }
      break;
  }
  
  // Update dynamic data
  dynamic_data[current_step].positioning = positioning;
  
  // Update streak
  if (positioning == prev_position && positioning != 0) {
    dynamic_data[current_step].streak = dynamic_data[current_step-1].streak + 1;
  } else {
    dynamic_data[current_step].streak = positioning != 0 ? 1 : 0;
  }
  
  // Calculate unrealized P&L
  if (positioning != 0) {
    // Simple P&L calculation based on spread change
    float spread_change = data[current_step].spread_zscore - data[current_step-1].spread_zscore;
    float position_value = positioning * spread_change; // Positive if position direction matches spread movement
    dynamic_data[current_step].unrealized = dynamic_data[current_step-1].unrealized + position_value;
  } else {
    dynamic_data[current_step].unrealized = 0.0f;
  }
}

float Environment::calculate_reward()
{
  // Calculating reward based on P&L change and penalizing frequent trading
  float pnl_change = dynamic_data[current_step].unrealized - dynamic_data[current_step-1].unrealized;
  float realized_profit = dynamic_data[current_step].realized;
  
  // Combine unrealized and realized profits
  float reward = pnl_change + realized_profit;
  
  // Penalize excessive position changes (optional)
  if (dynamic_data[current_step].positioning != dynamic_data[current_step-1].positioning) {
    reward -= 0.01f; // Small trading cost/penalty
  }
  
  return reward;
}

UpdateData Environment::forward(int action)
{
  UpdateData uData;
  
  // Execute action
  execute(action);
  
  // Calculate reward
  float reward = calculate_reward();
  total_reward += reward;
  
  // Update step
  current_step++;
  
  // Check if episode is done
  bool done = current_step >= data_size - 1;
  
  // Prepare return data
  uData.state = current_state();
  uData.reward = reward;
  uData.done = done;
  
  return uData;
}