#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

#include "common.hpp"
#include "minja/chat-template.hpp"
#include "minja/minja.hpp"
#include <cassert>
#include <chrono>
#include <llama/ggml.h>
#include <llama/llama.h>
#include <lloyal/nlohmann/json.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file helpers.hpp
 * @brief Helper Utilities
 *
 * Collection of utility functions for common llama.cpp operations:
 * - Batch operations: Build and manage token batches for decoding
 * - Chat template processing: Format messages, extract stop tokens, validate templates
 * - Parameter conversion: KV cache type mapping, string validation helpers
 * - String utilities: Repeat, join, split operations
 *
 * Source: Vendored from llama.cpp/common/
 * License: MIT License - Copyright (c) 2023-2024 The ggml.ai team
 */

// Forward declarations for detail namespace (defined at end of file)
namespace lloyal::detail {
std::string common_token_to_piece(const struct llama_vocab *vocab,
                                  llama_token token, bool special);
std::string get_token_safe(const llama_model *model, llama_token token);
const char *get_chatml_template();
std::string apply_chat_template_helper(const std::string &template_str,
                                       const nlohmann::ordered_json &messages,
                                       const std::string &bos_token,
                                       const std::string &eos_token,
                                       bool add_generation_prompt,
                                       bool add_bos, bool add_eos);
} // namespace lloyal::detail

namespace lloyal {

// Forward declare for chat template
using json = nlohmann::ordered_json;

// ===== BATCH UTILITIES =====

/**
 * @brief Clear batch to empty state
 *
 * Resets the batch token counter to prepare for new tokens.
 * Does not deallocate buffer memory.
 *
 * @param batch Batch to clear (modified in place)
 *
 * @note Only resets n_tokens counter, buffer capacity remains unchanged
 */
inline void batch_clear(llama_batch &batch) { batch.n_tokens = 0; }

/**
 * @brief Add single token to batch with position and sequence info
 *
 * Appends a token to the batch at the current n_tokens position, then increments
 * the counter. Assigns position embedding, sequence IDs, and logits flag.
 *
 * @param batch Batch to modify (appends token at batch.n_tokens)
 * @param id Token ID to add
 * @param pos Position embedding for this token (e.g., 0, 1, 2...)
 * @param seq_ids Sequence IDs this token belongs to (usually single-element vector {0})
 * @param logits Whether to compute logits for this token
 * @param capacity Optional capacity check for DEBUG builds (default: -1 disables check)
 *
 * @warning Caller must ensure batch has sufficient capacity (n_tokens < n_max)
 *          to avoid buffer overflow. No runtime bounds checking in release builds.
 *
 * @note DEBUG builds enable capacity assertion if capacity > 0
 */
inline void batch_add(llama_batch &batch, llama_token id, int32_t pos,
                      const std::vector<llama_seq_id> &seq_ids, bool logits,
                      int32_t capacity = -1) {
// Debug bounds checking to prevent buffer overflows
#ifdef DEBUG
  if (capacity > 0) {
    assert(batch.n_tokens < capacity && "batch_add: token capacity exceeded");
  }
#endif

  const auto i = batch.n_tokens;
  batch.token[i] = id;
  batch.pos[i] = pos;
  batch.n_seq_id[i] = static_cast<int32_t>(seq_ids.size());
  for (size_t j = 0; j < seq_ids.size(); ++j) {
    batch.seq_id[i][j] = seq_ids[j];
  }
  batch.logits[i] = logits ? 1 : 0;
  batch.n_tokens++;
}

// ===== CHAT TEMPLATE TYPES (PUBLIC API) =====

/**
 * @brief Result from complete chat template processing
 *
 * Contains formatted prompt and dynamically detected stop tokens specific
 * to the model's chat template (ChatML, Llama-3, etc.).
 */
struct ChatTemplateResult {
  std::string prompt;                      ///< Formatted chat prompt ready for tokenization
  std::vector<std::string> additional_stops; ///< Template-specific stop tokens (e.g., "<|im_end|>", "<|eot_id|>")
};

/**
 * @brief Format chat messages using model's built-in template
 *
 * Applies chat template (Jinja2) to format message array into a single prompt string.
 * Automatically queries model metadata for BOS/EOS tokens and add_bos/add_eos flags.
 *
 * Template selection hierarchy:
 * 1. template_override (if provided)
 * 2. model's embedded template (from GGUF metadata)
 * 3. ChatML fallback (default)
 *
 * @param model Llama model (can be null, will use ChatML fallback)
 * @param messages_json JSON array of messages: [{"role":"user","content":"..."},...]
 * @param template_override Optional Jinja2 template string (default: empty, uses model template)
 * @return Formatted prompt string ready for tokenization
 *
 * @throws std::exception if JSON parsing fails (caught internally, returns empty string)
 *
 * @note Strips BOS/EOS wrapper tokens if model metadata indicates they're added during tokenization
 *       to prevent double-token issues
 */
inline std::string
format_chat_template_from_model(const llama_model *model,
                                const std::string &messages_json,
                                const std::string &template_override = "") {
  try {
    json messages = json::parse(messages_json);

    // Determine template source
    std::string template_str;
    if (!template_override.empty()) {
      template_str = template_override;
    } else if (model) {
      const char *model_template = llama_model_chat_template(model, nullptr);
      if (model_template && strlen(model_template) > 0) {
        template_str = model_template;
      }
    }

    if (template_str.empty()) {
      template_str = detail::get_chatml_template();
    }

    // Get BOS/EOS tokens and metadata from model
    std::string bos_token, eos_token;
    bool add_bos = false, add_eos = false;

    if (model) {
      const auto *vocab = llama_model_get_vocab(model);
      bos_token = detail::get_token_safe(model, llama_vocab_bos(vocab));
      eos_token = detail::get_token_safe(model, llama_vocab_eos(vocab));

      // Query GGUF metadata to determine if wrapper tokens should be stripped
      // (they'll be re-added during tokenization if the model expects them)
      add_bos = llama_vocab_get_add_bos(vocab);
      add_eos = llama_vocab_get_add_eos(vocab);
    }

    return detail::apply_chat_template_helper(template_str, messages, bos_token,
                                              eos_token, true, add_bos, add_eos);

  } catch (const std::exception &e) {
    return "";
  }
}

/**
 * @brief Dynamically detect stop tokens from chat template
 *
 * Analyzes template string to identify template-specific stop tokens and verifies
 * they exist in the model's vocabulary. Prevents generating invalid tokens that
 * would cause tokenization failures.
 *
 * Supported patterns:
 * - ChatML: <|im_end|>, <|endoftext|> (when template contains "im_start")
 * - Llama-3: <|eom_id|>, <|eot_id|> (when template contains "eom_id" or "eot_id")
 * - Fallback: Model's EOT token from vocabulary
 *
 * @param model Llama model (can be null, returns empty vector)
 * @param template_str Jinja2 template string to analyze
 * @return Vector of stop token strings that exist in model vocabulary
 *
 * @note Only returns tokens that successfully tokenize to single token IDs.
 *       Prevents returning strings that would split into multiple tokens.
 */
inline std::vector<std::string>
extract_template_stop_tokens(const llama_model *model,
                             const std::string &template_str) {
  std::vector<std::string> stops;

  if (!model)
    return stops;

  const auto *vocab = llama_model_get_vocab(model);
  if (!vocab)
    return stops;

  // Check what tokens actually exist in this model's vocabulary
  const auto get_token_if_exists =
      [&](const std::string &token_text) -> std::string {
    std::vector<llama_token> tokens(1);
    int n_tokens = llama_tokenize(vocab, token_text.c_str(),
                                  static_cast<int32_t>(token_text.length()),
                                  tokens.data(), 1, false, true);
    if (n_tokens == 1) {
      return token_text;
    }
    return "";
  };

  // For ChatML-style templates
  if (template_str.find("im_start") != std::string::npos) {
    auto token = get_token_if_exists("<|im_end|>");
    if (!token.empty())
      stops.push_back(token);

    token = get_token_if_exists("<|endoftext|>");
    if (!token.empty())
      stops.push_back(token);
  }

  // For Llama-3 style templates
  if (template_str.find("eom_id") != std::string::npos ||
      template_str.find("eot_id") != std::string::npos) {
    auto token = get_token_if_exists("<|eom_id|>");
    if (!token.empty())
      stops.push_back(token);

    token = get_token_if_exists("<|eot_id|>");
    if (!token.empty())
      stops.push_back(token);
  }

  // Always check for model's EOT token as fallback
  auto eot_token = llama_vocab_eot(vocab);
  if (eot_token != LLAMA_TOKEN_NULL) {
    std::string eot_text =
        detail::common_token_to_piece(vocab, eot_token, true);
    if (!eot_text.empty() &&
        std::find(stops.begin(), stops.end(), eot_text) == stops.end()) {
      stops.push_back(eot_text);
    }
  }

  // Fallback to EOS when no template-specific stops found
  // Covers Zephyr-style templates that use eos_token (</s>) as the stop signal
  if (stops.empty()) {
    auto eos_token = llama_vocab_eos(vocab);
    if (eos_token != LLAMA_TOKEN_NULL) {
      std::string eos_text =
          detail::common_token_to_piece(vocab, eos_token, true);
      if (!eos_text.empty()) {
        stops.push_back(eos_text);
      }
    }
  }

  return stops;
}

/**
 * @brief Complete chat template processing with stop token detection
 *
 * Combines format_chat_template_from_model() and extract_template_stop_tokens()
 * into a single call for convenience. Returns both formatted prompt and detected
 * stop tokens.
 *
 * @param model Llama model (can be null, will use ChatML fallback)
 * @param messages_json JSON array of messages: [{"role":"user","content":"..."},...]
 * @param template_override Optional Jinja2 template string (default: empty, uses model template)
 * @return ChatTemplateResult with formatted prompt and additional_stops vector
 *
 * @note Equivalent to calling format_chat_template_from_model() followed by
 *       extract_template_stop_tokens(), but more efficient as it only queries
 *       model metadata once.
 */
inline ChatTemplateResult
format_chat_template_complete(const llama_model *model,
                              const std::string &messages_json,
                              const std::string &template_override = "") {
  ChatTemplateResult result;

  try {
    json messages = json::parse(messages_json);

    std::string template_str;
    if (!template_override.empty()) {
      template_str = template_override;
    } else if (model) {
      const char *model_template = llama_model_chat_template(model, nullptr);
      if (model_template && strlen(model_template) > 0) {
        template_str = model_template;
      }
    }

    if (template_str.empty()) {
      template_str = detail::get_chatml_template();
    }

    std::string bos_token, eos_token;
    bool add_bos = false, add_eos = false;

    if (model) {
      const auto *vocab = llama_model_get_vocab(model);
      bos_token = detail::get_token_safe(model, llama_vocab_bos(vocab));
      eos_token = detail::get_token_safe(model, llama_vocab_eos(vocab));

      // Query GGUF metadata to determine if wrapper tokens should be stripped
      // (they'll be re-added during tokenization if the model expects them)
      add_bos = llama_vocab_get_add_bos(vocab);
      add_eos = llama_vocab_get_add_eos(vocab);
    }

    result.prompt = detail::apply_chat_template_helper(
        template_str, messages, bos_token, eos_token, true, add_bos, add_eos);
    result.additional_stops = extract_template_stop_tokens(model, template_str);

  } catch (const std::exception &e) {
    result.prompt = "";
    result.additional_stops.clear();
  }

  return result;
}

/**
 * @brief Validate chat template syntax
 *
 * Attempts to parse Jinja2 template string using minja engine to check for
 * syntax errors before usage.
 *
 * @param template_str Jinja2 template string to validate
 * @return True if template syntax is valid, false if parsing failed
 *
 * @note Uses empty BOS/EOS tokens for validation - only checks syntax, not semantics
 */
inline bool validate_chat_template_helper(const std::string &template_str) {
  try {
    minja::chat_template tmpl(template_str, "", "");
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

// ===== PARAMETER CONVERSION HELPERS =====

/**
 * @brief Get list of supported KV cache types
 *
 * Returns static vector of ggml_type enums representing supported quantization
 * formats for KV cache. Includes full-precision (F32, F16, BF16) and quantized
 * formats (Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1).
 *
 * @return Reference to static vector of supported cache types
 *
 * @note Returns const reference to avoid allocation on each call
 */
inline const std::vector<ggml_type> &get_kv_cache_types() {
  static const std::vector<ggml_type> types = {
      GGML_TYPE_F32,    GGML_TYPE_F16,  GGML_TYPE_BF16,
      GGML_TYPE_Q8_0,   GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
      GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
  };
  return types;
}

/**
 * @brief Convert cache type string to ggml_type enum
 *
 * Maps type name string (e.g., "f16", "q4_0") to corresponding ggml_type enum.
 * Used for parsing user-provided cache type configuration.
 *
 * @param s Type name string (e.g., "f16", "q4_0", "q8_0")
 * @return Matching ggml_type enum value
 * @throws std::runtime_error if type name is not in supported types list
 */
inline ggml_type kv_cache_type_from_str(const std::string &s) {
  const auto &kv_cache_types = get_kv_cache_types();
  for (const auto &type : kv_cache_types) {
    if (ggml_type_name(type) == s) {
      return type;
    }
  }
  throw std::runtime_error("Unsupported cache type: " + s);
}

/**
 * @brief Check if string represents a truthy value
 *
 * @param value String to check
 * @return True if value is "on", "enabled", "1", or "true"
 */
inline bool is_truthy(const std::string &value) {
  return value == "on" || value == "enabled" || value == "1" || value == "true";
}

/**
 * @brief Check if string represents a falsey value
 *
 * @param value String to check
 * @return True if value is "off", "disabled", "0", or "false"
 */
inline bool is_falsey(const std::string &value) {
  return value == "off" || value == "disabled" || value == "0" ||
         value == "false";
}

/**
 * @brief Check if string represents an auto value
 *
 * @param value String to check
 * @return True if value is "auto" or "-1"
 */
inline bool is_autoy(const std::string &value) {
  return value == "auto" || value == "-1";
}

// ===== STRING UTILITIES =====

// Repeat string n times
inline std::string string_repeat(const std::string &str, size_t n) {
  if (n == 0) {
    return "";
  }

  std::string result;
  result.reserve(str.length() * n);

  for (size_t i = 0; i < n; ++i) {
    result += str;
  }

  return result;
}

// Join strings with separator
inline std::string string_join(const std::vector<std::string> &values,
                               const std::string &separator) {
  std::ostringstream result;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      result << separator;
    }
    result << values[i];
  }
  return result.str();
}

// Split string by delimiter
inline std::vector<std::string> string_split(const std::string &str,
                                             const std::string &delimiter) {
  std::vector<std::string> parts;
  size_t start = 0;
  size_t end = str.find(delimiter);

  while (end != std::string::npos) {
    parts.push_back(str.substr(start, end - start));
    start = end + delimiter.length();
    end = str.find(delimiter, start);
  }

  parts.push_back(str.substr(start));

  return parts;
}

} // namespace lloyal

namespace lloyal::detail {

// ===== INTERNAL TOKEN HELPERS =====

// Token conversion helper
inline std::string common_token_to_piece(const struct llama_vocab *vocab,
                                         llama_token token, bool special) {
  std::string piece;
  piece.resize(
      piece.capacity()); // using string internal cache, 15 bytes + '\n'
  const int n_chars =
      llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
  if (n_chars < 0) {
    piece.resize(-n_chars);
    int check =
        llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    assert(check == -n_chars);
  } else {
    piece.resize(n_chars);
  }
  return piece;
}

// Extract token from vocabulary safely
inline std::string get_token_safe(const llama_model *model, llama_token token) {
  if (!model || token == LLAMA_TOKEN_NULL) {
    return "";
  }
  const auto *vocab = llama_model_get_vocab(model);
  return common_token_to_piece(vocab, token, /* special */ true);
}

// ===== INTERNAL TEMPLATE HELPERS =====

// Default ChatML template fallback
inline const char *get_chatml_template() {
  return "{% for message in messages %}"
         "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
         "'<|im_end|>' + '\\n'}}"
         "{% endfor %}"
         "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif "
         "%}";
}

// Apply chat template using minja engine (requires minja.hpp to be included
// first)
// Implements a round-trip pattern: template renders with wrapper tokens,
// then strips them conditionally based on metadata so they can be re-added
// during tokenization if the model expects them.
inline std::string apply_chat_template_helper(
    const std::string &template_str, const json &messages,
    const std::string &bos_token = "", const std::string &eos_token = "",
    bool add_generation_prompt = true, bool add_bos = false,
    bool add_eos = false) {
  try {
    // Create minja chat template with correct 3-parameter constructor
    minja::chat_template tmpl(template_str, bos_token, eos_token);

    // Prepare template inputs
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.tools = json::array(); // No tools for basic implementation
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.now = std::chrono::system_clock::now();

    // Apply template with default options (use_bos_token=true, use_eos_token=true)
    // This ensures template variables like {{ bos_token }} and {{ eos_token }}
    // remain available for templates to use as delimiters between messages.
    minja::chat_template_options opts;
    auto result = tmpl.apply(inputs, opts);

    // Conditional wrapper token stripping
    // Only strip wrapper tokens at start/end if the model's metadata indicates
    // they will be re-added during tokenization. This prevents double-token issues
    // while keeping template variables available for use as delimiters.
    if (add_bos && !bos_token.empty() && result.starts_with(bos_token)) {
      result = result.substr(bos_token.length());
    }
    if (add_eos && !eos_token.empty() && result.ends_with(eos_token)) {
      result = result.substr(0, result.length() - eos_token.length());
    }

    return result;
  } catch (const std::exception &e) {
    return "";
  }
}

} // namespace lloyal::detail
