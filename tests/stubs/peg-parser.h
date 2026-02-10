// Stub for llama.cpp common/peg-parser.h
// Provides minimal common_peg_arena type needed by chat.h

#pragma once

#include <string>
#include <vector>

// Minimal stub — only the types referenced by chat.h and chat_out.hpp
// Real PEG parsing is not exercised in unit tests (integration tests use real llama.cpp)

class common_peg_arena {
public:
  bool empty() const { return data_.empty(); }
  void load(const std::string& data) { data_ = data; }
  std::string save() const { return data_; }
private:
  std::string data_;
};
