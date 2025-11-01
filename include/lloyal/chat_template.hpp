#pragma once

#include "common.hpp"
#include "helpers.hpp"
#include <llama/llama.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

/**
 * Chat Template Orchestration Layer (Header-Only)
 *
 * Purpose: Wraps helpers.hpp chat template functions with fallback error handling.
 * NOT just re-exports - adds orchestration and fallback logic.
 *
 * Architecture:
 * - Uses public functions from helpers.hpp (format_chat_template_complete, validate_chat_template_helper)
 * - Adds fallback to simple "role: content" format when template processing fails
 * - Provides clean API for chat template formatting
 */

namespace lloyal::chat_template {

/**
 * Result from chat template formatting
 * SOURCE: ChatTemplate.h:24-28
 * NOTE: Named FormatResult, NOT ChatTemplateResult
 */
struct FormatResult {
  std::string prompt;                     // Formatted prompt text
  std::vector<std::string> additional_stops;  // Stop tokens from template
};

/**
 * Format chat messages using model's chat template with fallback
 * SOURCE: ChatTemplate.h:51-55
 *
 * Orchestration logic:
 * 1. Calls format_chat_template_complete() from helpers.hpp
 * 2. If template processing fails (empty prompt), falls back to simple format
 * 3. Handles JSON parsing errors
 *
 * Fallback hierarchy:
 * 1. template_override (if provided)
 * 2. Model's built-in template
 * 3. ChatML template
 * 4. Simple "role: content" format (this layer adds this)
 *
 * @param model Llama model (for template and vocab)
 * @param messages_json JSON string with messages array
 * @param template_override Optional custom template
 * @return FormatResult with formatted prompt and stop tokens
 */
inline FormatResult format(
  const llama_model* model,
  const std::string& messages_json,
  const std::string& template_override = ""
) {
  FormatResult result;

  try {
    // Step 1: Call helpers.hpp function for template processing
    // (This handles template selection, BOS/EOS tokens, and stop token extraction)
    ChatTemplateResult helper_result = format_chat_template_complete(
      model,
      messages_json,
      template_override
    );

    // Step 2: Check if template processing succeeded
    if (helper_result.prompt.empty()) {
      LLOYAL_LOG_DEBUG("[chat_template::format] Template processing failed, using fallback");

      // Step 3: Fallback to simple "role: content" format
      try {
        using json = nlohmann::ordered_json;
        json messages = json::parse(messages_json);
        std::string fallback;
        for (const auto& msg : messages) {
          if (msg.contains("role") && msg.contains("content")) {
            fallback += msg["role"].get<std::string>() + ": " +
                       msg["content"].get<std::string>() + "\n";
          }
        }

        result.prompt = fallback;
        result.additional_stops = {};  // No stop tokens for fallback

        LLOYAL_LOG_DEBUG("[chat_template::format] Using fallback format (%zu bytes)", fallback.size());
        return result;

      } catch (const std::exception& e) {
        LLOYAL_LOG_DEBUG("[chat_template::format] ERROR: Failed to parse messages JSON: %s", e.what());
        result.prompt = "";
        result.additional_stops = {};
        return result;
      }
    }

    // Step 4: Success - return formatted result
    result.prompt = helper_result.prompt;
    result.additional_stops = helper_result.additional_stops;

    LLOYAL_LOG_DEBUG("[chat_template::format] Successfully formatted with %zu stop tokens",
              result.additional_stops.size());
    return result;

  } catch (const std::exception& e) {
    LLOYAL_LOG_DEBUG("[chat_template::format] ERROR: %s", e.what());
    result.prompt = "";
    result.additional_stops = {};
    return result;
  }
}

/**
 * Validate chat template syntax
 * SOURCE: ChatTemplate.h:68
 *
 * Calls validate_chat_template_helper() from helpers.hpp.
 * Does NOT require a model (syntax-only validation).
 *
 * @param template_str Template string to validate
 * @return True if syntax is valid, false otherwise (never throws)
 */
inline bool validate(const std::string& template_str) {
  try {
    // Call helpers.hpp validation function
    bool isValid = validate_chat_template_helper(template_str);
    LLOYAL_LOG_DEBUG("[chat_template::validate] Template validation: %s", isValid ? "valid" : "invalid");
    return isValid;
  } catch (const std::exception& e) {
    LLOYAL_LOG_DEBUG("[chat_template::validate] ERROR: %s", e.what());
    return false;
  }
}

} // namespace lloyal::chat_template
