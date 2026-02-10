// Stub for llama.cpp common/chat.h
// Provides chat template types and functions used by chat_template.hpp

#pragma once

#include "llama/llama.h"
#include "llama_stubs.h"
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// Template storage (needs to be complete type before make_unique is called)
struct common_chat_templates {};

// Smart pointer type for templates
using common_chat_templates_ptr = std::unique_ptr<common_chat_templates>;

// Tool call structure
struct common_chat_tool_call {
  std::string name;
  std::string arguments;
  std::string id;
};

// Content part structure
struct common_chat_msg_content_part {
  std::string type;
  std::string text;
};

// Chat message structure - matches llama.cpp common/chat.h
struct common_chat_msg {
  std::string role;
  std::string content;
  std::vector<common_chat_msg_content_part> content_parts;
  std::vector<common_chat_tool_call> tool_calls;
  std::string reasoning_content;
  std::string tool_name;
  std::string tool_call_id;
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

// Parse JSON messages to common_chat_msg vector
// Stub implementation that extracts all fields from OpenAI-compatible JSON
inline std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(
    const nlohmann::ordered_json& messages
) {
  std::vector<common_chat_msg> result;

  for (const auto& msg : messages) {
    common_chat_msg chat_msg;

    if (msg.contains("role")) {
      chat_msg.role = msg["role"].get<std::string>();
    }

    if (msg.contains("content")) {
      const auto& content = msg["content"];
      if (content.is_null()) {
        chat_msg.content = "";
      } else if (content.is_string()) {
        chat_msg.content = content.get<std::string>();
      } else if (content.is_array()) {
        // Content parts array
        for (const auto& part : content) {
          common_chat_msg_content_part cp;
          cp.type = part.value("type", "text");
          cp.text = part.value("text", "");
          chat_msg.content_parts.push_back(cp);
        }
      }
    }

    if (msg.contains("tool_calls")) {
      for (const auto& tc : msg["tool_calls"]) {
        common_chat_tool_call tool_call;
        if (tc.contains("function")) {
          const auto& func = tc["function"];
          tool_call.name = func.value("name", "");
          if (func.contains("arguments")) {
            const auto& args = func["arguments"];
            tool_call.arguments = args.is_string() ? args.get<std::string>() : args.dump();
          }
        }
        if (tc.contains("id")) {
          tool_call.id = tc["id"].get<std::string>();
        }
        chat_msg.tool_calls.push_back(tool_call);
      }
    }

    if (msg.contains("reasoning_content")) {
      chat_msg.reasoning_content = msg["reasoning_content"].get<std::string>();
    }

    if (msg.contains("name")) {
      chat_msg.tool_name = msg["name"].get<std::string>();
    }

    if (msg.contains("tool_call_id")) {
      chat_msg.tool_call_id = msg["tool_call_id"].get<std::string>();
    }

    result.push_back(chat_msg);
  }

  return result;
}
