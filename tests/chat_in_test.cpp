/**
 * Unit tests for chat input formatting
 *
 * Tests the chat_in API using llama.cpp's common library:
 * - Template formatting with fallback
 * - Template validation
 * - Turn separator extraction
 * - Tools, tool_choice, reasoning_format passthrough
 *
 * Note: This test uses stubs since we can't load real models in unit tests.
 * For full integration testing, see chat_in_integration_test.cpp
 */

#include <doctest/doctest.h>
#include <lloyal/chat_in.hpp>
#include <nlohmann/json.hpp>
#include "llama_stubs.h"

using json = nlohmann::ordered_json;

// ===== HELPER FUNCTIONS =====

void resetTestConfig() {
  resetStubConfig();
}

// ===== BASIC FORMATTING TESTS =====

TEST_CASE("ChatIn: format returns FormatResult with prompt") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  }).dump();

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(result.prompt.find("Hello") != std::string::npos);
}

TEST_CASE("ChatIn: format with empty messages") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array().dump();

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(result.additional_stops.empty());
}

TEST_CASE("ChatIn: format with null model") {
  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Test"}}
  }).dump();

  auto result = lloyal::chat_in::format(nullptr, inputs);

  CHECK(result.prompt.find("Test") != std::string::npos);
}

TEST_CASE("ChatIn: format with invalid JSON") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = "invalid json";

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(result.prompt.empty());
}

// ===== VALIDATION TESTS =====

TEST_CASE("ChatIn: validate with valid template") {
  std::string valid_template = "{{ messages[0]['content'] }}";

  bool is_valid = lloyal::chat_in::validate(valid_template);

  CHECK(is_valid);
}

TEST_CASE("ChatIn: validate with invalid template") {
  std::string invalid_template = "{{ unclosed";

  bool is_valid = lloyal::chat_in::validate(invalid_template);

  CHECK(!is_valid);
}

// ===== MULTI-MESSAGE TESTS =====

TEST_CASE("ChatIn: format multi-turn conversation") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "First message"}},
    {{"role", "assistant"}, {"content", "First response"}},
    {{"role", "user"}, {"content", "Second message"}}
  }).dump();

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(result.prompt.find("First message") != std::string::npos);
  CHECK(result.prompt.find("First response") != std::string::npos);
  CHECK(result.prompt.find("Second message") != std::string::npos);
}

// ===== EDGE CASES =====

TEST_CASE("ChatIn: format with very long message") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  std::string long_content(10000, 'x');
  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", long_content}}
  }).dump();

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find(long_content) != std::string::npos);
}

TEST_CASE("ChatIn: format with special characters") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Quote: \"Hello\"\nNewline\tTab"}}
  }).dump();

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(!result.prompt.empty());
}

TEST_CASE("ChatIn: format with unicode content") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello 世界 🌍"}}
  }).dump();

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("世界") != std::string::npos);
}

// ===== FALLBACK_TO_EOG TESTS =====

TEST_CASE("ChatIn: fallback_to_eog with null model") {
  auto result = lloyal::chat_in::fallback_to_eog(nullptr);

  CHECK(result.empty());
}

// ===== TOOLS TESTS =====

TEST_CASE("ChatIn: format with tools returns non-default format") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What's the weather?"}}
  }).dump();
  inputs.tools_json = json::array({
    {{"type", "function"}, {"function", {
      {"name", "get_weather"},
      {"description", "Get weather"},
      {"parameters", {{"type", "object"}, {"properties", {{"location", {{"type", "string"}}}}}}}
    }}}
  }).dump();

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(!result.prompt.empty());
  // Stub sets format=GENERIC when tools are provided
  CHECK(result.format == COMMON_CHAT_FORMAT_GENERIC);
  CHECK(!result.grammar.empty());
}

TEST_CASE("ChatIn: format with tool_choice required") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Call a tool"}}
  }).dump();
  inputs.tools_json = json::array({
    {{"type", "function"}, {"function", {
      {"name", "test_tool"},
      {"description", "Test"},
      {"parameters", {{"type", "object"}}}
    }}}
  }).dump();
  inputs.tool_choice = "required";

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.format != COMMON_CHAT_FORMAT_CONTENT_ONLY);
}

TEST_CASE("ChatIn: format with tool_choice none") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Don't use tools"}}
  }).dump();
  inputs.tools_json = json::array({
    {{"type", "function"}, {"function", {
      {"name", "test_tool"},
      {"description", "Test"},
      {"parameters", {{"type", "object"}}}
    }}}
  }).dump();
  inputs.tool_choice = "none";

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(!result.prompt.empty());
}

TEST_CASE("ChatIn: format with invalid tools JSON falls back gracefully") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  }).dump();
  inputs.tools_json = "not valid json";

  // Should fall back to simple format rather than crash
  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(result.prompt.find("Hello") != std::string::npos);
}

// ===== REASONING FORMAT TESTS =====

TEST_CASE("ChatIn: format with reasoning_format auto") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Think about this"}}
  }).dump();
  inputs.reasoning_format = "auto";

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.reasoning_format == COMMON_REASONING_FORMAT_AUTO);
}

TEST_CASE("ChatIn: format with reasoning_format deepseek") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Think deeply"}}
  }).dump();
  inputs.reasoning_format = "deepseek";

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK);
}

TEST_CASE("ChatIn: format with enable_thinking false") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "No thinking please"}}
  }).dump();
  inputs.enable_thinking = false;

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(!result.prompt.empty());
}

// ===== FORMAT RESULT FIELD TESTS =====

TEST_CASE("ChatIn: format with add_generation_prompt false omits assistant prefix") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello"}},
    {{"role", "assistant"}, {"content", "Hi there"}}
  }).dump();

  // With generation prompt (default)
  auto with_gen = lloyal::chat_in::format(&model, inputs);
  CHECK(with_gen.prompt.find("assistant: ") != std::string::npos);

  // Without generation prompt
  inputs.add_generation_prompt = false;
  auto without_gen = lloyal::chat_in::format(&model, inputs);
  CHECK(!without_gen.prompt.empty());
  // Should NOT end with "assistant: " generation prompt
  CHECK(without_gen.prompt.find("assistant: Hi there") != std::string::npos);
  // Prompt without gen should be a prefix of prompt with gen
  CHECK(with_gen.prompt.substr(0, without_gen.prompt.size()) == without_gen.prompt);
}

TEST_CASE("ChatIn: format returns default format when no tools") {
  resetTestConfig();

  llama_model model{};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  }).dump();

  auto result = lloyal::chat_in::format(&model, inputs);

  CHECK(result.format == COMMON_CHAT_FORMAT_CONTENT_ONLY);
  CHECK(result.grammar.empty());
  CHECK(result.grammar_triggers.empty());
  CHECK(result.preserved_tokens.empty());
  CHECK(result.parser.empty());
  CHECK(!result.grammar_lazy);
  CHECK(!result.thinking_forced_open);
}
