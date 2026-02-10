// Stub for llama.cpp common/chat.h
// Provides chat template types and functions used by chat_in.hpp and chat_out.hpp

#pragma once

#include "common.h"
#include "peg-parser.h"
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

// Tool choice enum
enum common_chat_tool_choice {
  COMMON_CHAT_TOOL_CHOICE_AUTO,
  COMMON_CHAT_TOOL_CHOICE_REQUIRED,
  COMMON_CHAT_TOOL_CHOICE_NONE,
};

// Chat format enum (from llama.cpp common/chat.h)
enum common_chat_format {
  COMMON_CHAT_FORMAT_CONTENT_ONLY,
  COMMON_CHAT_FORMAT_GENERIC,
  COMMON_CHAT_FORMAT_MISTRAL_NEMO,
  COMMON_CHAT_FORMAT_MAGISTRAL,
  COMMON_CHAT_FORMAT_LLAMA_3_X,
  COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
  COMMON_CHAT_FORMAT_DEEPSEEK_R1,
  COMMON_CHAT_FORMAT_FIREFUNCTION_V2,
  COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2,
  COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
  COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
  COMMON_CHAT_FORMAT_HERMES_2_PRO,
  COMMON_CHAT_FORMAT_COMMAND_R7B,
  COMMON_CHAT_FORMAT_GRANITE,
  COMMON_CHAT_FORMAT_GPT_OSS,
  COMMON_CHAT_FORMAT_SEED_OSS,
  COMMON_CHAT_FORMAT_NEMOTRON_V2,
  COMMON_CHAT_FORMAT_APERTUS,
  COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS,
  COMMON_CHAT_FORMAT_GLM_4_5,
  COMMON_CHAT_FORMAT_MINIMAX_M2,
  COMMON_CHAT_FORMAT_KIMI_K2,
  COMMON_CHAT_FORMAT_QWEN3_CODER_XML,
  COMMON_CHAT_FORMAT_APRIEL_1_5,
  COMMON_CHAT_FORMAT_XIAOMI_MIMO,
  COMMON_CHAT_FORMAT_SOLAR_OPEN,
  COMMON_CHAT_FORMAT_EXAONE_MOE,
  COMMON_CHAT_FORMAT_PEG_SIMPLE,
  COMMON_CHAT_FORMAT_PEG_NATIVE,
  COMMON_CHAT_FORMAT_PEG_CONSTRUCTED,
  COMMON_CHAT_FORMAT_COUNT,
};

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

// Tool definition structure
struct common_chat_tool {
  std::string name;
  std::string description;
  std::string parameters;  // JSON schema string
};

// Template inputs (all fields from llama.cpp)
struct common_chat_templates_inputs {
  std::vector<common_chat_msg> messages;
  std::string grammar;
  std::string json_schema;
  bool add_generation_prompt = true;
  bool use_jinja = true;
  std::vector<common_chat_tool> tools;
  common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
  bool parallel_tool_calls = false;
  common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
  bool enable_thinking = true;
};

// Template output parameters (all fields from llama.cpp)
struct common_chat_params {
  common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
  std::string prompt;
  std::string grammar;
  bool grammar_lazy = false;
  bool thinking_forced_open = false;
  std::vector<common_grammar_trigger> grammar_triggers;
  std::vector<std::string> preserved_tokens;
  std::vector<std::string> additional_stops;
  std::string parser;
};

// Parser parameters (for common_chat_parse)
struct common_chat_parser_params {
  common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
  common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
  bool reasoning_in_content = false;
  bool thinking_forced_open = false;
  bool parse_tool_calls = true;
  common_peg_arena parser = {};

  common_chat_parser_params() = default;
  common_chat_parser_params(const common_chat_params& chat_params) {
    format = chat_params.format;
    thinking_forced_open = chat_params.thinking_forced_open;
  }
};

// ===== STUB IMPLEMENTATIONS =====

inline common_chat_templates_ptr common_chat_templates_init(
    const llama_model* model,
    const std::string& template_override
) {
  (void)model;
  (void)template_override;
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

  // When tools are provided, simulate format detection
  if (!inputs.tools.empty()) {
    params.format = COMMON_CHAT_FORMAT_GENERIC;
    // Stub: generate a simple grammar string to indicate tools were processed
    params.grammar = "root ::= content | tool_call";
  }

  return params;
}

inline bool common_chat_verify_template(
    const std::string& template_str,
    bool use_jinja
) {
  (void)use_jinja;
  size_t open_expr = 0, close_expr = 0;
  size_t open_stmt = 0, close_stmt = 0;

  for (size_t i = 0; i + 1 < template_str.size(); ++i) {
    if (template_str[i] == '{' && template_str[i+1] == '{') open_expr++;
    if (template_str[i] == '}' && template_str[i+1] == '}') close_expr++;
    if (template_str[i] == '{' && template_str[i+1] == '%') open_stmt++;
    if (template_str[i] == '%' && template_str[i+1] == '}') close_stmt++;
  }

  bool has_jinja = (open_expr > 0 || open_stmt > 0);
  bool balanced = (open_expr == close_expr) && (open_stmt == close_stmt);

  return has_jinja && balanced;
}

// Parse JSON messages to common_chat_msg vector
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

// Parse tools from OpenAI-compatible JSON
inline std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(
    const nlohmann::ordered_json& tools
) {
  std::vector<common_chat_tool> result;
  for (const auto& tool : tools) {
    common_chat_tool t;
    if (tool.contains("function")) {
      const auto& func = tool["function"];
      t.name = func.value("name", "");
      t.description = func.value("description", "");
      if (func.contains("parameters")) {
        t.parameters = func["parameters"].dump();
      }
    }
    result.push_back(t);
  }
  return result;
}

// Parse tool choice string to enum
inline common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(
    const std::string& choice
) {
  if (choice == "required") return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
  if (choice == "none") return COMMON_CHAT_TOOL_CHOICE_NONE;
  return COMMON_CHAT_TOOL_CHOICE_AUTO;
}

// Parse reasoning format from string
inline common_reasoning_format common_reasoning_format_from_name(
    const std::string& name
) {
  if (name == "auto") return COMMON_REASONING_FORMAT_AUTO;
  if (name == "deepseek") return COMMON_REASONING_FORMAT_DEEPSEEK;
  if (name == "deepseek_legacy") return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
  return COMMON_REASONING_FORMAT_NONE;
}

// Get format name as string
inline const char* common_chat_format_name(common_chat_format format) {
  switch (format) {
    case COMMON_CHAT_FORMAT_CONTENT_ONLY: return "content_only";
    case COMMON_CHAT_FORMAT_GENERIC: return "generic";
    case COMMON_CHAT_FORMAT_MISTRAL_NEMO: return "mistral_nemo";
    case COMMON_CHAT_FORMAT_LLAMA_3_X: return "llama_3_x";
    case COMMON_CHAT_FORMAT_DEEPSEEK_R1: return "deepseek_r1";
    case COMMON_CHAT_FORMAT_HERMES_2_PRO: return "hermes_2_pro";
    default: return "unknown";
  }
}

// Parse model output — stub returns content passthrough
inline common_chat_msg common_chat_parse(
    const std::string& input,
    bool is_partial,
    const common_chat_parser_params& syntax
) {
  (void)is_partial;
  (void)syntax;
  common_chat_msg result;
  result.role = "assistant";
  result.content = input;
  return result;
}
