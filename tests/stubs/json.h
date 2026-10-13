// Stub for llama.cpp common/json.h
// The thin JSON value llama.cpp's common library accepts at its entry points.
// Here it is nlohmann::ordered_json behind the same two names liblloyal uses:
// common_json::parse and common_json_error.

#pragma once

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

struct common_json_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class common_json {
public:
  common_json() = default;
  explicit common_json(nlohmann::ordered_json value) : value_(std::move(value)) {}

  static common_json parse(const std::string& text) {
    try {
      return common_json(nlohmann::ordered_json::parse(text));
    } catch (const nlohmann::json::parse_error& e) {
      throw common_json_error(e.what());
    }
  }

  const nlohmann::ordered_json& raw() const { return value_; }

private:
  nlohmann::ordered_json value_;
};
