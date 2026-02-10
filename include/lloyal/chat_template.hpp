#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

/**
 * @file chat_template.hpp
 * @brief Chat Template Formatting and Turn Separator Extraction
 *
 * Provides high-level chat template processing with graceful fallback handling.
 * This module orchestrates template formatting and extracts turn separator tokens
 * for warm multi-turn continuation.
 *
 * @section arch Architecture
 *
 * Uses llama.cpp's common library (chat.h) for template processing:
 * - common_chat_templates_init(): Initialize templates from model
 * - common_chat_templates_apply(): Apply template to messages
 * - Graceful degradation when template processing fails
 * - Turn separator extraction for warm prefill parity
 *
 * @section deps Dependencies
 * - common/chat.h: common_chat_templates_init, common_chat_templates_apply
 * - tokenizer.hpp: tokenize(), detokenize(), is_eog()
 *
 * @section fallback Fallback Hierarchy
 * 1. template_override (if provided)
 * 2. Model's built-in template (llama_model_chat_template)
 * 3. ChatML template (default fallback in llama.cpp)
 * 4. Simple "role: content" format (last resort)
 *
 * @see common_chat_templates_init()
 * @see lloyal::tokenizer
 */

#include "common.hpp"
#include "tokenizer.hpp"
#include <llama/llama.h>
#include <chat.h>         // llama.cpp common library: common_chat_templates_*
#include <nlohmann/json.hpp>
#include <algorithm>
#include <string>
#include <vector>

namespace lloyal::chat_template {

/**
 * @brief Result from chat template formatting
 *
 * Contains the formatted prompt text and any additional stop tokens
 * extracted from the template. Named FormatResult (not ChatTemplateResult)
 * to distinguish from other types.
 */
struct FormatResult {
  std::string prompt;                        ///< Formatted prompt text ready for tokenization
  std::vector<std::string> additional_stops; ///< Stop tokens extracted from template (e.g., "<|eot_id|>")
};

/**
 * @brief Format chat messages using model's chat template with fallback
 *
 * Orchestrates chat template processing with graceful degradation:
 * 1. Attempts common_chat_templates_apply() from llama.cpp common
 * 2. Falls back to simple "role: content" format if template fails
 * 3. Returns empty result on JSON parsing errors (never throws)
 *
 * @param model Llama model pointer (provides template and vocabulary)
 * @param messages_json JSON string containing messages array, e.g.:
 *        `[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]`
 * @param template_override Optional Jinja2-style template string. If empty,
 *        uses model's built-in template or falls back to ChatML.
 *
 * @return FormatResult containing:
 *         - prompt: Formatted text ready for tokenization
 *         - additional_stops: Stop tokens from template (may be empty)
 *
 * @note This function never throws. On error, returns empty prompt.
 *
 * @see common_chat_templates_apply()
 *
 * @code
 * // Basic usage (cold start)
 * auto result = chat_template::format(model, R"([{"role":"user","content":"Hi"}])");
 * const auto* vocab = llama_model_get_vocab(model);
 * auto tokens = tokenizer::tokenize(vocab, result.prompt, true, true);
 * @endcode
 */
inline FormatResult format(const llama_model *model,
                           const std::string &messages_json,
                           const std::string &template_override = "") {
  FormatResult result;

  try {
    // Parse JSON messages
    using json = nlohmann::ordered_json;
    json messages_array = json::parse(messages_json);

    // Initialize templates from model (or override)
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, template_override);
    if (!tmpls) {
      LLOYAL_LOG_DEBUG("[chat_template::format] Template init failed, using fallback");
      goto fallback;
    }

    {
      // Parse JSON to common_chat_msg using llama.cpp's parser
      // This preserves ALL message fields: tool_calls, name, tool_call_id,
      // reasoning_content, content_parts - required for proper Jinja template rendering
      std::vector<common_chat_msg> messages = common_chat_msgs_parse_oaicompat(messages_array);

      // Prepare template inputs
      common_chat_templates_inputs inputs;
      inputs.messages = messages;
      inputs.add_generation_prompt = true;
      inputs.use_jinja = true;

      // Apply template
      common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);

      result.prompt = params.prompt;
      result.additional_stops = params.additional_stops;

      LLOYAL_LOG_DEBUG(
          "[chat_template::format] Successfully formatted with %zu stop tokens",
          result.additional_stops.size());
      return result;
    }

  } catch (const std::exception &e) {
    LLOYAL_LOG_DEBUG("[chat_template::format] Template processing failed: %s", e.what());
  }

fallback:
  // Fallback to simple "role: content" format
  try {
    using json = nlohmann::ordered_json;
    json messages = json::parse(messages_json);
    std::string fallback_prompt;
    for (const auto &msg : messages) {
      if (msg.contains("role") && msg.contains("content")) {
        std::string role = msg["role"].get<std::string>();
        std::string content;
        const auto& c = msg["content"];
        if (c.is_null()) {
          content = "";
        } else if (c.is_string()) {
          content = c.get<std::string>();
        } else {
          content = c.dump();
        }
        fallback_prompt += role + ": " + content + "\n";
      }
    }

    result.prompt = fallback_prompt;
    result.additional_stops = {}; // No stop tokens for fallback

    LLOYAL_LOG_DEBUG(
        "[chat_template::format] Using fallback format (%zu bytes)",
        fallback_prompt.size());
    return result;

  } catch (const std::exception &e) {
    LLOYAL_LOG_DEBUG(
        "[chat_template::format] ERROR: Failed to parse messages JSON: %s",
        e.what());
    result.prompt = "";
    result.additional_stops = {};
    return result;
  }
}

/**
 * @brief Validate chat template syntax
 *
 * Performs syntax-only validation of a Jinja2-style chat template.
 * Does NOT require a model — useful for validating user-provided templates
 * before attempting to format messages.
 *
 * @param template_str Jinja2-style template string to validate
 *
 * @return `true` if template syntax is valid, `false` otherwise
 *
 * @note This function never throws. Returns false on any error.
 *
 * @see common_chat_verify_template()
 *
 * @code
 * // Validate before using
 * if (chat_template::validate(user_template)) {
 *   auto result = chat_template::format(model, messages, user_template);
 * } else {
 *   // Fall back to model's default template
 *   auto result = chat_template::format(model, messages);
 * }
 * @endcode
 */
inline bool validate(const std::string &template_str) {
  try {
    // Use llama.cpp's common library validation
    bool isValid = common_chat_verify_template(template_str, /* use_jinja */ true);
    LLOYAL_LOG_DEBUG("[chat_template::validate] Template validation: %s",
                     isValid ? "valid" : "invalid");
    return isValid;
  } catch (const std::exception &e) {
    LLOYAL_LOG_DEBUG("[chat_template::validate] ERROR: %s", e.what());
    return false;
  }
}

/**
 * @brief Get EOG token as fallback when template parsing fails
 *
 * Returns the model's end-of-generation token wrapped in a vector.
 * Prefers EOT (end-of-turn) token, falling back to EOS (end-of-sequence).
 *
 * @param model Llama model pointer
 *
 * @return Vector containing single EOG token, or empty vector if no EOG token exists
 *
 * @note Internal helper function. Not intended for direct use.
 *
 * @see get_turn_separator()
 */
inline std::vector<llama_token> fallback_to_eog(const llama_model* model) {
  if (model == nullptr) {
    return {};
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  llama_token eot = llama_vocab_eot(vocab);
  if (eot == LLAMA_TOKEN_NULL) {
    eot = llama_vocab_eos(vocab);
  }
  if (eot != LLAMA_TOKEN_NULL) {
    return {eot};
  }
  return {};
}

/**
 * @brief Get token text safely
 *
 * @param model Llama model pointer
 * @param token Token ID
 * @return Token text, or empty string if invalid
 */
inline std::string get_token_safe(const llama_model *model, llama_token token) {
  if (!model || token == LLAMA_TOKEN_NULL) {
    return "";
  }
  return lloyal::tokenizer::detokenize(model, token);
}

/**
 * @brief Get turn separator tokens for the model's chat template
 *
 * Extracts the token sequence that closes an assistant turn and transitions
 * to the next message. This enables exact parity between cold-start and
 * warm multi-turn continuation paths.
 *
 * @section algorithm Algorithm
 *
 * Uses a 3-message probe technique:
 * 1. Format: `[user:"X", assistant:SENTINEL, user:SENTINEL2]`
 * 2. Extract text between SENTINEL and SENTINEL2
 * 3. Tokenize with `parse_special=true`
 * 4. Keep tokens up to and including EOG + trailing whitespace
 *
 * @section examples Template-Specific Results
 *
 * | Template | Separator Tokens | Text Representation |
 * |----------|------------------|---------------------|
 * | ChatML   | `[im_end, \n]`   | `<|im_end|>\n`      |
 * | Llama-3  | `[eot_id]`       | `<|eot_id|>`        |
 * | Phi-3    | `[end, \n]`      | `<|end|>\n`         |
 * | Zephyr   | `[eos, \n]`      | `</s>\n`            |
 *
 * @param model Llama model pointer (provides template and vocabulary)
 *
 * @return Vector of token IDs representing the turn separator.
 *         Falls back to single EOG token if template parsing fails.
 *         Returns empty vector only if model has no EOG token.
 *
 * @note Result is typically cached by the caller (e.g., SessionContext).
 *
 * @note Uses sentinel strings with ASCII control characters for extraction.
 *       Collision is theoretically possible but practically impossible. If
 *       extraction fails or produces invalid results (no EOG token), the
 *       function safely falls back to returning the model's EOG token.
 *
 * @see lloyal::tokenizer::is_eog()
 *
 * @code
 * // Warm multi-turn continuation with exact cold/warm parity
 * auto separator = chat_template::get_turn_separator(model);
 * const auto* vocab = llama_model_get_vocab(model);
 * std::string delta_prompt = format_new_turn(messages);  // Your formatted new turn
 * auto delta_tokens = tokenizer::tokenize(vocab, delta_prompt, false, true);
 *                                      // add_bos=false: continuing, not fresh start
 *
 * // Prepend separator to delta for exact match with cold path
 * std::vector<llama_token> prefill_tokens;
 * prefill_tokens.insert(prefill_tokens.end(), separator.begin(), separator.end());
 * prefill_tokens.insert(prefill_tokens.end(), delta_tokens.begin(), delta_tokens.end());
 *
 * branch.prefill(prefill_tokens);
 * @endcode
 */
inline std::vector<llama_token> get_turn_separator(const llama_model* model) {
  using json = nlohmann::ordered_json;

  if (!model) return {};

  // Collision-resistant sentinels
  const std::string SENTINEL = "\x1F__LLOYAL_SEP__\x1F";
  const std::string SENTINEL2 = "\x1F__LLOYAL_SEP2__\x1F";

  try {
    // Initialize templates from model
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    if (!tmpls) {
      return fallback_to_eog(model);
    }

    // 3-message probe: captures REAL assistant→user boundary
    std::vector<common_chat_msg> messages = {
      {.role = "user", .content = "X"},
      {.role = "assistant", .content = SENTINEL},
      {.role = "user", .content = SENTINEL2}
    };

    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.add_generation_prompt = false;  // Don't add assistant prompt at end
    inputs.use_jinja = true;

    auto params = common_chat_templates_apply(tmpls.get(), inputs);
    const std::string& formatted = params.prompt;

    // Extract substring between sentinels
    size_t sep_start = formatted.rfind(SENTINEL);
    if (sep_start == std::string::npos) {
      return fallback_to_eog(model);
    }
    sep_start += SENTINEL.length();

    size_t sep_end = formatted.find(SENTINEL2, sep_start);
    if (sep_end == std::string::npos) {
      return fallback_to_eog(model);
    }

    std::string between = formatted.substr(sep_start, sep_end - sep_start);
    if (between.empty()) {
      return fallback_to_eog(model);
    }

    // Tokenize with parse_special=true
    const auto* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = lloyal::tokenizer::tokenize(vocab, between, false, true);
    if (tokens.empty()) {
      return fallback_to_eog(model);
    }

    // Extract: everything up to and including EOG + trailing whitespace
    std::vector<llama_token> separator;
    bool found_eog = false;

    for (auto tok : tokens) {
      if (!found_eog) {
        separator.push_back(tok);
        if (lloyal::tokenizer::is_eog(model, tok)) {
          found_eog = true;
        }
      } else {
        // After EOG, only keep whitespace tokens
        // NOTE: Only ASCII whitespace is checked. Unicode whitespace (NBSP, zero-width,
        // etc.) is not handled because chat templates universally use ASCII whitespace
        // for turn boundaries. If a template used Unicode whitespace, those tokens would
        // be treated as the next message opener and excluded from the separator.
        std::string text = lloyal::tokenizer::detokenize(model, tok);
        bool is_whitespace = !text.empty() && std::all_of(text.begin(), text.end(),
            [](unsigned char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; });
        if (is_whitespace) {
          separator.push_back(tok);
        } else {
          break;  // Non-whitespace = next message opener, stop
        }
      }
    }

    if (separator.empty() || !found_eog) {
      return fallback_to_eog(model);
    }

    return separator;

  } catch (const std::exception& e) {
    LLOYAL_LOG_DEBUG("[chat_template::get_turn_separator] Error: %s", e.what());
    return fallback_to_eog(model);
  }
}

} // namespace lloyal::chat_template
