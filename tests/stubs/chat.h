// Stub for llama.cpp common/chat.h
// Provides chat template types and functions used by chat_template.hpp

#pragma once

#include "llama/llama.h"
#include "llama_stubs.h"
#include <memory>
#include <string>
#include <vector>

// Forward declaration
struct common_chat_templates;

// Smart pointer type for templates
using common_chat_templates_ptr = std::unique_ptr<common_chat_templates>;

// Chat message structure
struct common_chat_msg {
  std::string role;
  std::string content;
};

// Template inputs
struct common_chat_templates_inputs {
  std::vector<common_chat_msg> messages;
  bool add_generation_prompt = true;
  bool use_jinja = true;
};

// Template output parameters
struct common_chat_params {
  std::string prompt;
  std::vector<std::string> additional_stops;
};

// Stub implementations
inline common_chat_templates_ptr common_chat_templates_init(
    const llama_model* model,
    const std::string& template_override
) {
  (void)model;
  (void)template_override;
  // Return a valid pointer so null checks pass
  return std::make_unique<common_chat_templates>();
}

inline common_chat_params common_chat_templates_apply(
    const common_chat_templates* tmpls,
    const common_chat_templates_inputs& inputs
) {
  (void)tmpls;
  common_chat_params params;

  // Simple stub: concatenate messages as "role: content\n"
  for (const auto& msg : inputs.messages) {
    params.prompt += msg.role + ": " + msg.content + "\n";
  }

  if (inputs.add_generation_prompt) {
    params.prompt += "assistant: ";
  }

  return params;
}

inline bool common_chat_verify_template(
    const std::string& template_str,
    bool use_jinja
) {
  (void)use_jinja;
  // Simple validation: check for balanced Jinja syntax
  // Count opening and closing braces
  size_t open_expr = 0, close_expr = 0;
  size_t open_stmt = 0, close_stmt = 0;

  for (size_t i = 0; i + 1 < template_str.size(); ++i) {
    if (template_str[i] == '{' && template_str[i+1] == '{') open_expr++;
    if (template_str[i] == '}' && template_str[i+1] == '}') close_expr++;
    if (template_str[i] == '{' && template_str[i+1] == '%') open_stmt++;
    if (template_str[i] == '%' && template_str[i+1] == '}') close_stmt++;
  }

  // Must have balanced braces and at least some Jinja syntax
  bool has_jinja = (open_expr > 0 || open_stmt > 0);
  bool balanced = (open_expr == close_expr) && (open_stmt == close_stmt);

  return has_jinja && balanced;
}

// Empty struct definition (only needs to exist for unique_ptr)
struct common_chat_templates {};
