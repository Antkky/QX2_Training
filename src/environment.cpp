#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

struct UpdateData
{
  torch::Tensor state;
  float reward;
  bool done;
};

struct ImportData {
  string time;
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

struct DynamicData {
  int streak;
  int positioning;
  float unrealized;
  float realized;
};

class Environment
{
private:
  vector<ImportData> data;
  vector<DynamicData> dynamic_data;
  int current_step;
  int window_size;
  int data_size;
  int positioning;
  float total_reward;
  float total_profit;
  const int state_dim = 17;
  
  void execute(int action);
  float calculate_reward();
  Tensor current_state();
  bool parse_csv_line(const string& line, ImportData& data);
  bool load_data(const string& filepath);

public:
  Environment(string filepath, int window_size);
  ~Environment();
  UpdateData forward(int action);
  Tensor reset();
  int get_state_dim() const { return state_dim; }
};

Environment::Environment(string filepath, int window_size) : window_size(window_size), current_step(0), positioning(0), total_reward(0.0f), total_profit(0.0f)
{
  if (!load_data(filepath)) {
    cerr << "Failed to load data from " << filepath << endl;
    throw runtime_error("Data loading failed");
  }
  
  data_size = data.size();
  
  dynamic_data.resize(data_size);
  for (auto& dd : dynamic_data) {
    dd.streak = 0;
    dd.positioning = 0;
    dd.unrealized = 0.0f;
    dd.realized = 0.0f;
  }
  
  cout << "Environment initialized with " << data_size << " data points" << endl;
}

Environment::~Environment()
{
}

bool Environment::parse_csv_line(const string& line, ImportData& data_point) {
  stringstream ss(line);
  string cell;
  
  // Parse time
  if (!getline(ss, cell, ',')) return false;
  data_point.time = cell;
  
  // Parse numeric values
  try {
    if (!getline(ss, cell, ',')) return false;
    data_point.asset1_price = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.asset2_price = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.ratio_price = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.hedge_ratio = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.spread_zscore = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.correlation = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.cointegration = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.rsi1 = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.rsi2 = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.rsi3 = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.macd1 = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.macd2 = stof(cell);
    
    if (!getline(ss, cell, ',')) return false;
    data_point.macd3 = stof(cell);
    
  } catch (const exception& e) {
    cerr << "Error parsing line: " << e.what() << endl;
    return false;
  }
  
  return true;
}

bool Environment::load_data(const string& filepath) {
  ifstream file(filepath);
  if (!file.is_open()) {
    cerr << "Could not open file: " << filepath << endl;
    return false;
  }
  
  string line;
  getline(file, line);
  
  while (getline(file, line)) {
    ImportData data_point;
    if (parse_csv_line(line, data_point)) {
      data.push_back(data_point);
    }
  }
  
  return !data.empty();
}

torch::Tensor Environment::reset()
{
  current_step = window_size;
  positioning = 0;
  total_reward = 0.0f;
  total_profit = 0.0f;

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
  vector<float> state_vec;

  cout << state_vec << endl;
  
  for (int i = current_step - window_size; i < current_step; i++) {
    state_vec.push_back(data[i].spread_zscore);
  }
  
  state_vec.push_back(data[current_step].spread_zscore);
  state_vec.push_back(data[current_step].correlation);
  state_vec.push_back(data[current_step].rsi1);
  state_vec.push_back(data[current_step].rsi2);
  state_vec.push_back(data[current_step].macd1);
  
  state_vec.push_back(static_cast<float>(positioning));
  state_vec.push_back(dynamic_data[current_step].unrealized);
  
  torch::Tensor state = torch::tensor(state_vec).reshape({1, 1, state_dim});
  
  return state;
}

void Environment::execute(int action)
{
  int prev_position = positioning;
  
  switch (action) {
    case 0:
      break;
    case 1:
      if (positioning != 1) {
        if (positioning == -1) {
          dynamic_data[current_step].realized = dynamic_data[current_step-1].unrealized;
          dynamic_data[current_step].streak = 0;
        }
        positioning = 1;
      }
      break;
    case 2:
      if (positioning != -1) {
        if (positioning == 1) {
          dynamic_data[current_step].realized = dynamic_data[current_step-1].unrealized;
          dynamic_data[current_step].streak = 0;
        }
        positioning = -1;
      }
      break;
    case 3:
      if (positioning != 0) {
        dynamic_data[current_step].realized = dynamic_data[current_step-1].unrealized;
        positioning = 0;
        dynamic_data[current_step].streak = 0;
      }
      break;
  }
  
  dynamic_data[current_step].positioning = positioning;
  
  if (positioning == prev_position && positioning != 0) {
    dynamic_data[current_step].streak = dynamic_data[current_step-1].streak + 1;
  } else {
    dynamic_data[current_step].streak = positioning != 0 ? 1 : 0;
  }
  
  if (positioning != 0) {
    float spread_change = data[current_step].spread_zscore - data[current_step-1].spread_zscore;
    float position_value = positioning * spread_change;
    dynamic_data[current_step].unrealized = dynamic_data[current_step-1].unrealized + position_value;
  } else {
    dynamic_data[current_step].unrealized = 0.0f;
  }
}

float Environment::calculate_reward()
{
  float pnl_change = dynamic_data[current_step].unrealized - dynamic_data[current_step-1].unrealized;
  float realized_profit = dynamic_data[current_step].realized;

  float reward = pnl_change + realized_profit;
  
  if (dynamic_data[current_step].positioning != dynamic_data[current_step-1].positioning) {
    reward -= 0.01f;
  }
  
  return reward;
}

UpdateData Environment::forward(int action)
{
  UpdateData uData;
  
  execute(action);

  float reward = calculate_reward();
  total_reward += reward;

  current_step++;
  
  bool done = current_step >= data_size - 1;
  
  uData.state = current_state();
  uData.reward = reward;
  uData.done = done;
  
  return uData;
}