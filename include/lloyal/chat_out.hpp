#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

/**
 * @file chat_out.hpp
 * @brief Chat Output Parsing
 *
 * Wraps llama.cpp's common_chat_parse() to extract structured content from
 * model output: plain text, reasoning/thinking content, and tool calls.
 * Stateless — each call is independent.
 *
 * @section usage Usage Pattern
 *
 * Typically paired with chat_in::format():
 * 1. chat_in::format() → FormatResult (includes format enum + reasoning_format)
 * 2. Generate tokens using the formatted prompt
 * 3. chat_out::parse() with the format from step 1 → ParseResult
 *
 * @section deps Dependencies
 * - common/chat.h: common_chat_parse()
 *
 * @see lloyal::chat_in for input formatting
 */

#include "common.hpp"
#include <llama/llama.h>
#include <chat.h>
#include <peg-parser.h>
#include <exception>
#include <string>
#include <vector>

/**
 * @brief Chat output parsing (tool calls, reasoning, content)
 *
 * Wraps llama.cpp's common_chat_parse() to extract structured content from
 * model output. Pairs with @ref lloyal::chat_in -- use the format from
 * chat_in::FormatResult to select the correct parser.
 */
namespace lloyal::chat_out {

/**
 * @brief A single tool call extracted from model output
 */
struct ToolCall {
  std::string name;        ///< Tool/function name
  std::string arguments;   ///< JSON string of arguments
  std::string id;          ///< Tool call ID (may be empty if model doesn't generate IDs)
};

/**
 * @brief Result from parsing model output
 */
struct ParseResult {
  std::string content;              ///< Main response text
  std::string reasoning_content;    ///< Extracted thinking/reasoning blocks
  std::vector<ToolCall> tool_calls; ///< Extracted tool calls (empty if none)
};

/**
 * @brief Parse model output with explicit format
 *
 * Uses the format detected by chat_in::format() to apply the correct parser.
 * For most formats, this delegates to common_chat_parse() which handles
 * 25+ model-specific output formats (DeepSeek, Mistral, Hermes, etc.).
 *
 * @param output The raw model output text to parse
 * @param format The chat format (from chat_in::FormatResult.format)
 * @param reasoning_format How to handle reasoning/thinking blocks
 * @param is_partial True if output is incomplete (streaming)
 * @param thinking_forced_open Whether thinking tag was forced open
 * @param parser_data Serialized PEG parser (from chat_in::FormatResult.parser).
 *                    Required for PEG format models; ignored for others.
 *
 * @return ParseResult with content, reasoning_content, and tool_calls
 *
 * @note This function never throws. On error, returns raw output as content.
 *
 * @warning For PEG format models (COMMON_CHAT_FORMAT_PEG_*), the @p parser_data
 * parameter must contain the serialized PEG parser from chat_in::FormatResult::parser.
 * Omitting it will cause parse failures for these formats.
 *
 * @see lloyal::chat_in::format()
 *
 * @code{.cpp}
 * auto fmt = chat_in::format(model, inputs);
 * // ... generate tokens ...
 * auto parsed = chat_out::parse(output_text, fmt.format, fmt.reasoning_format,
 *                                false, fmt.thinking_forced_open, fmt.parser);
 * if (!parsed.tool_calls.empty()) {
 *   // Handle tool calls
 * }
 * @endcode
 */
inline ParseResult parse(
    const std::string& output,
    common_chat_format format,
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE,
    bool is_partial = false,
    bool thinking_forced_open = false,
    const std::string& parser_data = ""
) {
  ParseResult result;

  try {
    // Build parser params
    common_chat_parser_params syntax;
    syntax.format = format;
    syntax.reasoning_format = reasoning_format;
    syntax.thinking_forced_open = thinking_forced_open;

    // Load serialized PEG parser if provided (required for PEG format models)
    if (!parser_data.empty()) {
      syntax.parser.load(parser_data);
    }

    // Call llama.cpp's output parser
    common_chat_msg msg = common_chat_parse(output, is_partial, syntax);

    // Convert to ParseResult
    result.content = msg.content;
    result.reasoning_content = msg.reasoning_content;

    for (const auto& tc : msg.tool_calls) {
      result.tool_calls.push_back({tc.name, tc.arguments, tc.id});
    }

  } catch (const std::exception& e) {
    LLOYAL_LOG_DEBUG("[chat_out::parse] Parse failed: %s, returning raw output", e.what());
    result.content = output;
  }

  return result;
}

/**
 * @brief Parse model output with auto-detected format from model template
 *
 * Convenience overload that detects the format from the model's template.
 * More expensive than the explicit-format overload since it initializes
 * templates and applies them to detect the format.
 *
 * @param model Llama model pointer
 * @param output The raw model output text to parse
 * @param is_partial True if output is incomplete (streaming)
 *
 * @return ParseResult with content, reasoning_content, and tool_calls
 *
 * @note Prefer the explicit-format overload when you already have a FormatResult.
 *
 * @see lloyal::chat_in::format()
 */
inline ParseResult parse(
    const llama_model* model,
    const std::string& output,
    bool is_partial = false
) {
  ParseResult result;

  try {
    // Init templates to detect format
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    if (!tmpls) {
      result.content = output;
      return result;
    }

    // Apply with empty messages to get format detection
    common_chat_templates_inputs inputs;
    inputs.messages = {{.role = "user", .content = ""}};
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;

    common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);

    // Delegate to explicit-format overload
    return parse(output, params.format, COMMON_REASONING_FORMAT_NONE, is_partial,
                 params.thinking_forced_open);

  } catch (const std::exception& e) {
    LLOYAL_LOG_DEBUG("[chat_out::parse] Auto-detect failed: %s", e.what());
    result.content = output;
    return result;
  }
}

} // namespace lloyal::chat_out
