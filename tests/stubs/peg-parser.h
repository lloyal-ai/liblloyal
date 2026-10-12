// Stub for llama.cpp common/peg-parser.h
// Provides minimal common_peg_arena type needed by chat.h

#pragma once

#include <string>

// Minimal stub: only what chat.h and chat_out.hpp touch. The real arena is an
// object graph that is built by the template engine and copied by value; it
// has no string form. Real PEG parsing is not exercised in unit tests
// (integration tests use real llama.cpp).
class common_peg_arena {
public:
  bool empty() const { return name_.empty(); }
  std::string dump() const { return name_; }

  // Stub-only: lets a test tell one arena from another
  void set_name(std::string name) { name_ = std::move(name); }

private:
  std::string name_;
};
