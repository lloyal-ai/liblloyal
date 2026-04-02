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
  std::string generation_prompt;                    ///< Generation prompt prefill (e.g. "<think>")
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

      // Implicit empty system prompt stripping: if messages[0] is {system, ""},
      // strip the resulting empty system block from the output. This lets callers
      // suppress template auto-injection (e.g. SmolLM2/ChatML) by prepending an
      // empty system message — the library completes the intent by removing the
      // rendered empty block, leaving only the user+assistant portion.
      if (!messages.empty() && messages[0].role == "system" && messages[0].content.empty()) {
        bool stripped = false;

        // Primary: format [{system:""}] to learn the empty system prefix
        try {
          common_chat_msg sys_msg;
          sys_msg.role = "system";
          sys_msg.content = "";

          common_chat_templates_inputs sys_inputs;
          sys_inputs.messages = {sys_msg};
          sys_inputs.add_generation_prompt = false;
          sys_inputs.use_jinja = true;
          auto sys_params = common_chat_templates_apply(tmpls.get(), sys_inputs);

          const auto& sys_prefix = sys_params.prompt;
          if (!sys_prefix.empty() &&
              params.prompt.size() >= sys_prefix.size() &&
              params.prompt.substr(0, sys_prefix.size()) == sys_prefix) {
            params.prompt = params.prompt.substr(sys_prefix.size());
            stripped = true;
            LLOYAL_LOG_DEBUG("[chat_in::format] Stripped empty system prefix (%zu bytes)", sys_prefix.size());
          }
        } catch (const std::exception &e) {
          LLOYAL_LOG_DEBUG("[chat_in::format] Primary stripping failed: %s", e.what());
        }

        // Sentinel fallback: template requires a user message (e.g. Qwen 3.5).
        // Format [{system:""}, {user:SENTINEL}] and [{user:SENTINEL}], subtract
        // to learn the empty system prefix.
        if (!stripped) {
          try {
            static const std::string SENTINEL = "\x1F__LLOYAL_SYS_STRIP__\x1F";

            common_chat_msg sys_msg;
            sys_msg.role = "system";
            sys_msg.content = "";
            common_chat_msg user_msg;
            user_msg.role = "user";
            user_msg.content = SENTINEL;

            common_chat_templates_inputs with_sys;
            with_sys.messages = {sys_msg, user_msg};
            with_sys.add_generation_prompt = false;
            with_sys.use_jinja = true;
            auto with_sys_params = common_chat_templates_apply(tmpls.get(), with_sys);

            common_chat_templates_inputs without_sys;
            without_sys.messages = {user_msg};
            without_sys.add_generation_prompt = false;
            without_sys.use_jinja = true;
            auto without_sys_params = common_chat_templates_apply(tmpls.get(), without_sys);

            const auto& with_prompt = with_sys_params.prompt;
            const auto& without_prompt = without_sys_params.prompt;

            // If with_sys ends with without_sys, the prefix is the difference
            if (with_prompt.size() > without_prompt.size() &&
                with_prompt.substr(with_prompt.size() - without_prompt.size()) == without_prompt) {
              std::string sys_prefix = with_prompt.substr(0, with_prompt.size() - without_prompt.size());
              if (!sys_prefix.empty() &&
                  params.prompt.size() >= sys_prefix.size() &&
                  params.prompt.substr(0, sys_prefix.size()) == sys_prefix) {
                params.prompt = params.prompt.substr(sys_prefix.size());
                LLOYAL_LOG_DEBUG("[chat_in::format] Stripped empty system prefix via sentinel (%zu bytes)", sys_prefix.size());
              }
            } else {
              LLOYAL_LOG_DEBUG("[chat_in::format] Sentinel subtraction failed, skipping strip");
            }
          } catch (const std::exception &e) {
            LLOYAL_LOG_DEBUG("[chat_in::format] Sentinel stripping also failed: %s", e.what());
          }
        }
      }

      // Populate ALL result fields from common_chat_params
      result.prompt = params.prompt;
      result.additional_stops = params.additional_stops;
      result.format = params.format;
      result.grammar = params.grammar;
      result.grammar_lazy = params.grammar_lazy;
      result.generation_prompt = params.generation_prompt;
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

    // Retry with synthetic user: templates like Qwen 3.5 require a user message.
    // Inject a sentinel user message, re-apply the template, then strip the
    // sentinel user turn from the output to recover just the system/tool portion.
    try {
      using json = nlohmann::ordered_json;
      json messages_array = json::parse(inputs.messages_json);

      common_chat_templates_ptr tmpls = common_chat_templates_init(model, inputs.template_override);
      if (tmpls) {
        static const std::string SENTINEL = "\x1F__LLOYAL_RETRY__\x1F";

        std::vector<common_chat_msg> messages = common_chat_msgs_parse_oaicompat(messages_array);

        // Check that no user role exists (otherwise the failure was something else)
        bool has_user = false;
        for (const auto& m : messages) {
          if (m.role == "user") { has_user = true; break; }
        }

        if (!has_user) {
          // Build augmented messages: original + synthetic user
          std::vector<common_chat_msg> augmented = messages;
          common_chat_msg sentinel_user;
          sentinel_user.role = "user";
          sentinel_user.content = SENTINEL;
          augmented.push_back(sentinel_user);

          common_chat_templates_inputs tmpl_inputs;
          tmpl_inputs.messages = augmented;
          tmpl_inputs.add_generation_prompt = false;
          tmpl_inputs.use_jinja = true;

          // Carry over tools so the system block includes tool definitions
          if (!inputs.tools_json.empty()) {
            json tools_array = json::parse(inputs.tools_json);
            tmpl_inputs.tools = common_chat_tools_parse_oaicompat(tools_array);
            tmpl_inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(inputs.tool_choice);
            tmpl_inputs.parallel_tool_calls = inputs.parallel_tool_calls;
          }
          tmpl_inputs.reasoning_format = common_reasoning_format_from_name(inputs.reasoning_format);
          tmpl_inputs.enable_thinking = inputs.enable_thinking;
          tmpl_inputs.json_schema = inputs.json_schema;
          tmpl_inputs.grammar = inputs.grammar;

          common_chat_params params = common_chat_templates_apply(tmpls.get(), tmpl_inputs);

          // Also format just [{user:SENTINEL}] to learn its rendered form
          common_chat_msg user_only_msg;
          user_only_msg.role = "user";
          user_only_msg.content = SENTINEL;

          common_chat_templates_inputs user_only_inputs;
          user_only_inputs.messages = {user_only_msg};
          user_only_inputs.add_generation_prompt = false;
          user_only_inputs.use_jinja = true;
          auto user_only_params = common_chat_templates_apply(tmpls.get(), user_only_inputs);

          const auto& full = params.prompt;
          const auto& user_suffix = user_only_params.prompt;

          // Strip the sentinel user turn from the end
          if (full.size() > user_suffix.size() &&
              full.substr(full.size() - user_suffix.size()) == user_suffix) {
            params.prompt = full.substr(0, full.size() - user_suffix.size());

            // Strip empty system block if messages[0] is {system, ""}
            if (!messages.empty() && messages[0].role == "system" && messages[0].content.empty()) {
              // Use sentinel subtraction: [{system:""}, {user:S}] minus [{user:S}]
              common_chat_msg sys_msg;  sys_msg.role = "system"; sys_msg.content = "";
              common_chat_msg usr_msg;  usr_msg.role = "user";   usr_msg.content = SENTINEL;

              common_chat_templates_inputs with_sys_inputs;
              with_sys_inputs.messages = {sys_msg, usr_msg};
              with_sys_inputs.add_generation_prompt = false;
              with_sys_inputs.use_jinja = true;

              common_chat_templates_inputs without_sys_inputs;
              without_sys_inputs.messages = {usr_msg};
              without_sys_inputs.add_generation_prompt = false;
              without_sys_inputs.use_jinja = true;

              try {
                auto with_sys = common_chat_templates_apply(tmpls.get(), with_sys_inputs);
                auto without_sys = common_chat_templates_apply(tmpls.get(), without_sys_inputs);
                if (with_sys.prompt.size() > without_sys.prompt.size() &&
                    with_sys.prompt.substr(with_sys.prompt.size() - without_sys.prompt.size()) == without_sys.prompt) {
                  std::string sys_prefix = with_sys.prompt.substr(0, with_sys.prompt.size() - without_sys.prompt.size());
                  if (!sys_prefix.empty() &&
                      params.prompt.size() >= sys_prefix.size() &&
                      params.prompt.substr(0, sys_prefix.size()) == sys_prefix) {
                    params.prompt = params.prompt.substr(sys_prefix.size());
                    LLOYAL_LOG_DEBUG("[chat_in::format] Retry: stripped empty system prefix (%zu bytes)", sys_prefix.size());
                  }
                }
              } catch (...) {
                // Stripping failed — proceed without it
              }
            }

            result.prompt = params.prompt;
            result.additional_stops = params.additional_stops;
            result.format = params.format;
            result.grammar = params.grammar;
            result.grammar_lazy = params.grammar_lazy;
            result.generation_prompt = params.generation_prompt;
            result.grammar_triggers = params.grammar_triggers;
            result.preserved_tokens = params.preserved_tokens;
            result.parser = params.parser;
            result.reasoning_format = tmpl_inputs.reasoning_format;

            LLOYAL_LOG_DEBUG(
                "[chat_in::format] Retry with synthetic user succeeded, format=%d (%zu bytes)",
                static_cast<int>(result.format), result.prompt.size());
            return result;
          } else {
            LLOYAL_LOG_DEBUG("[chat_in::format] Retry sentinel subtraction failed");
          }
        }
      }
    } catch (const std::exception &e2) {
      LLOYAL_LOG_DEBUG("[chat_in::format] Retry also failed: %s", e2.what());
    }
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

} // namespace lloyal::chat_in
