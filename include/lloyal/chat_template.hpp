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
 * This module wraps lower-level helpers and adds:
 * - Graceful degradation when template processing fails
 * - Turn separator extraction for warm prefill parity
 * - Clean FormatResult API for template formatting + stop token extraction
 *
 * @section deps Dependencies
 * - helpers.hpp: format_chat_template_complete(), validate_chat_template_helper()
 * - tokenizer.hpp: tokenize(), detokenize(), is_eog()
 *
 * @section fallback Fallback Hierarchy
 * 1. template_override (if provided)
 * 2. Model's built-in template (llama_model_chat_template)
 * 3. ChatML template (default)
 * 4. Simple "role: content" format (last resort)
 *
 * @see lloyal::detail::format_chat_template_complete()
 * @see lloyal::tokenizer
 */

#include "common.hpp"
#include "helpers.hpp"
#include "tokenizer.hpp"
#include <llama/llama.h>
#include "nlohmann/json.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace lloyal::chat_template {

/**
 * @brief Result from chat template formatting
 *
 * Contains the formatted prompt text and any additional stop tokens
 * extracted from the template. Named FormatResult (not ChatTemplateResult)
 * to distinguish from helpers.hpp types.
 */
struct FormatResult {
  std::string prompt;                        ///< Formatted prompt text ready for tokenization
  std::vector<std::string> additional_stops; ///< Stop tokens extracted from template (e.g., "<|eot_id|>")
};

/**
 * @brief Format chat messages using model's chat template with fallback
 *
 * Orchestrates chat template processing with graceful degradation:
 * 1. Attempts format_chat_template_complete() from helpers.hpp
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
 * @see lloyal::detail::format_chat_template_complete()
 *
 * @code
 * // Basic usage
 * auto result = chat_template::format(model, R"([{"role":"user","content":"Hi"}])");
 * const auto* vocab = llama_model_get_vocab(model);
 * auto tokens = tokenizer::tokenize(vocab, result.prompt, true, true);
 *
 * // With custom template
 * auto custom_result = chat_template::format(model, messages_json, custom_template);
 * @endcode
 */
inline FormatResult format(const llama_model *model,
                           const std::string &messages_json,
                           const std::string &template_override = "") {
  FormatResult result;

  try {
    // Step 1: Call helpers.hpp function for template processing
    // (This handles template selection, BOS/EOS tokens, and stop token
    // extraction)
    ChatTemplateResult helper_result =
        format_chat_template_complete(model, messages_json, template_override);

    // Step 2: Check if template processing succeeded
    if (helper_result.prompt.empty()) {
      LLOYAL_LOG_DEBUG(
          "[chat_template::format] Template processing failed, using fallback");

      // Step 3: Fallback to simple "role: content" format
      try {
        using json = nlohmann::ordered_json;
        json messages = json::parse(messages_json);
        std::string fallback;
        for (const auto &msg : messages) {
          if (msg.contains("role") && msg.contains("content")) {
            fallback += msg["role"].get<std::string>() + ": " +
                        msg["content"].get<std::string>() + "\n";
          }
        }

        result.prompt = fallback;
        result.additional_stops = {}; // No stop tokens for fallback

        LLOYAL_LOG_DEBUG(
            "[chat_template::format] Using fallback format (%zu bytes)",
            fallback.size());
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

    // Step 4: Success - return formatted result
    result.prompt = helper_result.prompt;
    result.additional_stops = helper_result.additional_stops;

    LLOYAL_LOG_DEBUG(
        "[chat_template::format] Successfully formatted with %zu stop tokens",
        result.additional_stops.size());
    return result;

  } catch (const std::exception &e) {
    LLOYAL_LOG_DEBUG("[chat_template::format] ERROR: %s", e.what());
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
 * @see lloyal::detail::validate_chat_template_helper()
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
    // Call helpers.hpp validation function
    bool isValid = validate_chat_template_helper(template_str);
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
 * | ChatML   | `[im_end, \n]`   | `<\|im_end\|>\n`    |
 * | Llama-3  | `[eot_id]`       | `<\|eot_id\|>`      |
 * | Phi-3    | `[end, \n]`      | `<\|end\|>\n`       |
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

  // Resolve template
  std::string template_str;
  const char* model_template = llama_model_chat_template(model, nullptr);
  if (model_template && strlen(model_template) > 0) {
    template_str = model_template;
  }
  if (template_str.empty()) {
    template_str = lloyal::detail::get_chatml_template();
  }

  // Extract BOS/EOS tokens
  const auto* vocab = llama_model_get_vocab(model);
  std::string bos_token = lloyal::detail::get_token_safe(model, llama_vocab_bos(vocab));
  std::string eos_token = lloyal::detail::get_token_safe(model, llama_vocab_eos(vocab));
  bool add_bos = llama_vocab_get_add_bos(vocab);

  // Collision-resistant sentinels
  const std::string SENTINEL = "\x1F__LLOYAL_SEP__\x1F";
  const std::string SENTINEL2 = "\x1F__LLOYAL_SEP2__\x1F";

  // 3-message probe: captures REAL assistant→user boundary
  json messages = json::array({
    {{"role", "user"}, {"content", "X"}},
    {{"role", "assistant"}, {"content", SENTINEL}},
    {{"role", "user"}, {"content", SENTINEL2}}
  });

  std::string formatted = lloyal::detail::apply_chat_template_helper(
      template_str, messages, bos_token, eos_token,
      false,    // add_generation_prompt=false
      add_bos,
      false     // add_eos=false — turn boundary, not sequence end
  );

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
}

} // namespace lloyal::chat_template
