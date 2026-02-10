#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

/**
 * @file chat_in.hpp
 * @brief Chat Input Formatting with Full Format Awareness
 *
 * Provides high-level chat template processing that passes through all
 * format-awareness fields (tools, grammar, reasoning) and returns all
 * output fields from common_chat_params. This enables callers to use
 * format-aware grammar constraining and output parsing.
 *
 * @section arch Architecture
 *
 * Uses llama.cpp's common library (chat.h) for template processing:
 * - common_chat_templates_init(): Initialize templates from model
 * - common_chat_templates_apply(): Apply template with full inputs
 * - common_chat_tools_parse_oaicompat(): Parse tool definitions
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
 * @see lloyal::chat_out for output parsing
 */

#include "common.hpp"
#include "tokenizer.hpp"
#include <llama/llama.h>
#include <chat.h>         // llama.cpp common library: common_chat_templates_*
#include <nlohmann/json.hpp>
#include <algorithm>
#include <exception>
#include <string>
#include <vector>

/**
 * @brief Chat input formatting with full format awareness
 *
 * Wraps llama.cpp's chat template engine to produce formatted prompts with
 * all format-awareness metadata (grammar, triggers, parser) needed for
 * correct output parsing via @ref lloyal::chat_out.
 */
namespace lloyal::chat_in {

/**
 * @brief Input parameters for chat formatting
 *
 * Controls all format-awareness fields passed to common_chat_templates_apply().
 * All fields have sensible defaults so callers can provide only what they need.
 */
struct FormatInputs {
  std::string messages_json;                       ///< JSON array of OpenAI-format messages (required)
  std::string template_override = "";              ///< Optional Jinja2 template override
  bool add_generation_prompt = true;               ///< Append assistant prompt prefix (set false for partial formatting)
  std::string tools_json = "";                     ///< JSON array of OpenAI-format tool definitions
  std::string tool_choice = "auto";                ///< "auto" | "required" | "none"
  bool parallel_tool_calls = false;                ///< Allow parallel tool calls
  std::string reasoning_format = "none";           ///< "none" | "auto" | "deepseek" | "deepseek_legacy"
  bool enable_thinking = true;                     ///< Enable <think> blocks (pairs with reasoning_format)
  std::string json_schema = "";                    ///< JSON schema for structured output
  std::string grammar = "";                        ///< Explicit GBNF grammar string
};

/**
 * @brief Result from chat template formatting with full format awareness
 *
 * Contains the formatted prompt and all fields from common_chat_params,
 * enabling format-aware grammar constraining and output parsing.
 */
struct FormatResult {
  // Core output
  std::string prompt;                              ///< Formatted prompt text ready for tokenization
  std::vector<std::string> additional_stops;       ///< Stop tokens extracted from template

  // Format awareness (all fields from common_chat_params)
  common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;  ///< Detected chat format
  std::string grammar;                             ///< GBNF grammar for constrained sampling
  bool grammar_lazy = false;                       ///< Whether grammar should use lazy compilation
  bool thinking_forced_open = false;               ///< Whether thinking tag is forced open
  std::vector<common_grammar_trigger> grammar_triggers; ///< Triggers for lazy grammar activation
  std::vector<std::string> preserved_tokens;       ///< Tokens to preserve during grammar constraining
  std::string parser;                              ///< PEG parser definition (for PEG formats)

  // Carried through for chat_out pairing
  common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE; ///< Reasoning format for output parsing
};

/**
 * @brief Format chat messages using model's chat template with full format awareness
 *
 * Orchestrates chat template processing with graceful degradation:
 * 1. Parses tools and messages from JSON
 * 2. Applies common_chat_templates_apply() with all format-awareness fields
 * 3. Returns all common_chat_params fields for downstream grammar/parsing use
 * 4. Falls back to simple "role: content" format if template fails
 * 5. Returns empty result on JSON parsing errors (never throws)
 *
 * @param model Llama model pointer (provides template and vocabulary)
 * @param inputs FormatInputs struct with messages, tools, and format options
 *
 * @return FormatResult containing prompt, format, grammar, triggers, and parser info
 *
 * @note This function never throws. On error, returns empty prompt.
 *
 * @see common_chat_templates_apply()
 * @see lloyal::chat_out::parse()
 *
 * @code{.cpp}
 * // Basic usage (no tools)
 * chat_in::FormatInputs inputs;
 * inputs.messages_json = R"([{"role":"user","content":"Hi"}])";
 * auto result = chat_in::format(model, inputs);
 * auto tokens = tokenizer::tokenize(vocab, result.prompt, true, true);
 *
 * // With tools
 * inputs.tools_json = R"([{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}])";
 * auto tool_result = chat_in::format(model, inputs);
 * // tool_result.format != CONTENT_ONLY when tools are active
 * // tool_result.grammar contains GBNF for constrained tool call output
 * @endcode
 */
inline FormatResult format(const llama_model *model, const FormatInputs& inputs) {
  FormatResult result;

  try {
    using json = nlohmann::ordered_json;
    json messages_array = json::parse(inputs.messages_json);

    // Initialize templates from model (or override)
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, inputs.template_override);
    if (!tmpls) {
      LLOYAL_LOG_DEBUG("[chat_in::format] Template init failed, using fallback");
      goto fallback;
    }

    {
      // Parse messages
      std::vector<common_chat_msg> messages = common_chat_msgs_parse_oaicompat(messages_array);

      // Build full template inputs
      common_chat_templates_inputs tmpl_inputs;
      tmpl_inputs.messages = messages;
      tmpl_inputs.add_generation_prompt = inputs.add_generation_prompt;
      tmpl_inputs.use_jinja = true;

      // Tools
      if (!inputs.tools_json.empty()) {
        json tools_array = json::parse(inputs.tools_json);
        tmpl_inputs.tools = common_chat_tools_parse_oaicompat(tools_array);
        tmpl_inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(inputs.tool_choice);
        tmpl_inputs.parallel_tool_calls = inputs.parallel_tool_calls;
      }

      // Reasoning
      tmpl_inputs.reasoning_format = common_reasoning_format_from_name(inputs.reasoning_format);
      tmpl_inputs.enable_thinking = inputs.enable_thinking;

      // Structured output
      tmpl_inputs.json_schema = inputs.json_schema;
      tmpl_inputs.grammar = inputs.grammar;

      // Apply template
      common_chat_params params = common_chat_templates_apply(tmpls.get(), tmpl_inputs);

      // Populate ALL result fields from common_chat_params
      result.prompt = params.prompt;
      result.additional_stops = params.additional_stops;
      result.format = params.format;
      result.grammar = params.grammar;
      result.grammar_lazy = params.grammar_lazy;
      result.thinking_forced_open = params.thinking_forced_open;
      result.grammar_triggers = params.grammar_triggers;
      result.preserved_tokens = params.preserved_tokens;
      result.parser = params.parser;

      // Carry reasoning_format through for chat_out pairing
      result.reasoning_format = tmpl_inputs.reasoning_format;

      LLOYAL_LOG_DEBUG(
          "[chat_in::format] Successfully formatted with format=%d, %zu stop tokens, grammar=%zu bytes",
          static_cast<int>(result.format),
          result.additional_stops.size(),
          result.grammar.size());
      return result;
    }

  } catch (const std::exception &e) {
    LLOYAL_LOG_DEBUG("[chat_in::format] Template processing failed: %s", e.what());
  }

fallback:
  // Fallback to simple "role: content" format
  try {
    using json = nlohmann::ordered_json;
    json messages = json::parse(inputs.messages_json);
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
    result.additional_stops = {};

    LLOYAL_LOG_DEBUG(
        "[chat_in::format] Using fallback format (%zu bytes)",
        fallback_prompt.size());
    return result;

  } catch (const std::exception &e) {
    LLOYAL_LOG_DEBUG(
        "[chat_in::format] ERROR: Failed to parse messages JSON: %s",
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
 * @return `true` if template syntax is valid, `false` otherwise
 * @note This function never throws. Returns false on any error.
 */
inline bool validate(const std::string &template_str) {
  try {
    bool isValid = common_chat_verify_template(template_str, /* use_jinja */ true);
    LLOYAL_LOG_DEBUG("[chat_in::validate] Template validation: %s",
                     isValid ? "valid" : "invalid");
    return isValid;
  } catch (const std::exception &e) {
    LLOYAL_LOG_DEBUG("[chat_in::validate] ERROR: %s", e.what());
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
 * @return Vector containing single EOG token, or empty vector if no EOG token exists
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
 * @code{.cpp}
 * auto separator = chat_in::get_turn_separator(model);
 * auto delta_tokens = tokenizer::tokenize(vocab, delta_prompt, false, true);
 * std::vector<llama_token> prefill_tokens;
 * prefill_tokens.insert(prefill_tokens.end(), separator.begin(), separator.end());
 * prefill_tokens.insert(prefill_tokens.end(), delta_tokens.begin(), delta_tokens.end());
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
    LLOYAL_LOG_DEBUG("[chat_in::get_turn_separator] Error: %s", e.what());
    return fallback_to_eog(model);
  }
}

/**
 * @brief Pre-tokenized wrapper tokens for warm multi-turn continuation
 *
 * Contains the three token sequences needed to inject a new user turn
 * into an existing conversation without re-formatting the full history.
 * Extracted from the model's chat template via sentinel probing.
 *
 * @section usage Usage (warm continuation)
 * @code{.cpp}
 * auto warm = chat_in::get_warm_turn_tokens(model);
 * auto content_tokens = tokenizer::tokenize(vocab, user_content, false, true);
 * std::vector<llama_token> prefill;
 * prefill.insert(prefill.end(), warm.turn_separator.begin(), warm.turn_separator.end());
 * prefill.insert(prefill.end(), warm.user_prefix.begin(), warm.user_prefix.end());
 * prefill.insert(prefill.end(), content_tokens.begin(), content_tokens.end());
 * prefill.insert(prefill.end(), warm.user_to_assistant.begin(), warm.user_to_assistant.end());
 * @endcode
 */
struct WarmTurnTokens {
  std::vector<llama_token> turn_separator;    ///< Closes previous assistant turn (e.g., [im_end, \n] or [eot_id])
  std::vector<llama_token> user_prefix;       ///< Opens new user turn (e.g., [im_start, "user", \n])
  std::vector<llama_token> user_to_assistant; ///< Closes user turn + opens assistant (e.g., [im_end, \n, im_start, "assistant", \n])
};

/**
 * @brief Extract warm turn wrapper tokens from model's chat template
 *
 * Uses sentinel probing to extract the pre-tokenized role wrappers needed
 * for warm multi-turn continuation. The warm path becomes:
 *   turn_separator + user_prefix + tokenize(content, false) + user_to_assistant
 *
 * This eliminates the need for full-conversation formatting and string diffing
 * on warm turns, fixing the BOS bug where tokenize(delta) adds a spurious BOS
 * on models with add_bos_token=true (e.g., Llama 3.2).
 *
 * @section algorithm Algorithm
 *
 * 1. Reuse get_turn_separator() for the separator tokens
 * 2. Probe with sentinels: [user:"X", assistant:S1, user:S2] + add_generation_prompt
 * 3. Extract user_prefix from text between S1 and S2, after stripping separator text
 * 4. Extract user_to_assistant from text after S2 to end of formatted output
 *
 * @param model Llama model pointer (provides template and vocabulary)
 * @return WarmTurnTokens with all three sequences populated.
 *         Returns empty struct if template parsing fails.
 *
 * @note Result is typically cached by the caller (e.g., SessionContext).
 */
inline WarmTurnTokens get_warm_turn_tokens(const llama_model* model) {
  WarmTurnTokens result;

  if (!model) return result;

  // Step 1: Get turn separator (reuse existing function)
  result.turn_separator = get_turn_separator(model);
  if (result.turn_separator.empty()) return result;

  // Step 2: Probe for user_prefix and user_to_assistant
  const std::string S1 = "\x1F__LLOYAL_W1__\x1F";
  const std::string S2 = "\x1F__LLOYAL_W2__\x1F";

  try {
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    if (!tmpls) return result;

    std::vector<common_chat_msg> messages = {
      {.role = "user",      .content = "X"},
      {.role = "assistant", .content = S1},
      {.role = "user",      .content = S2}
    };

    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.add_generation_prompt = true;  // Capture assistant prompt after user turn
    inputs.use_jinja = true;

    auto params = common_chat_templates_apply(tmpls.get(), inputs);
    const std::string& fmt = params.prompt;

    // Extract text between S1 and S2 (turn_sep + user_prefix)
    // and text after S2 (user_to_assistant)
    size_t s1_pos = fmt.rfind(S1);
    if (s1_pos == std::string::npos) return result;
    size_t s1_end = s1_pos + S1.length();

    size_t s2_pos = fmt.find(S2, s1_end);
    if (s2_pos == std::string::npos) return result;
    size_t s2_end = s2_pos + S2.length();

    std::string between_s1_s2 = fmt.substr(s1_end, s2_pos - s1_end);
    std::string after_s2 = fmt.substr(s2_end);

    // Strip turn separator text from between_s1_s2 to get user_prefix text
    std::string sep_text = tokenizer::detokenize_batch(model, result.turn_separator);
    std::string user_prefix_text = between_s1_s2;
    if (user_prefix_text.length() >= sep_text.length() &&
        user_prefix_text.substr(0, sep_text.length()) == sep_text) {
      user_prefix_text = user_prefix_text.substr(sep_text.length());
    }

    // Tokenize (no BOS, parse special tokens)
    const auto* vocab = llama_model_get_vocab(model);
    if (!vocab) return result;

    if (!user_prefix_text.empty()) {
      result.user_prefix = tokenizer::tokenize(vocab, user_prefix_text, false, true);
    }
    if (!after_s2.empty()) {
      result.user_to_assistant = tokenizer::tokenize(vocab, after_s2, false, true);
    }

    LLOYAL_LOG_DEBUG(
        "[chat_in::get_warm_turn_tokens] separator=%zu, user_prefix=%zu, user_to_assistant=%zu tokens",
        result.turn_separator.size(),
        result.user_prefix.size(),
        result.user_to_assistant.size());

    return result;

  } catch (const std::exception& e) {
    LLOYAL_LOG_DEBUG("[chat_in::get_warm_turn_tokens] Error: %s", e.what());
    return result;
  }
}

} // namespace lloyal::chat_in
