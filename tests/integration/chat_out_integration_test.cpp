/**
 * Integration tests for chat output parsing with real models
 *
 * Tests the chat_out API with actual GGUF models:
 * - Plain text output parsing
 * - Format round-trip with chat_in
 * - Model-based format auto-detection
 *
 * Requires: LLAMA_TEST_MODEL environment variable
 */

#include <doctest/doctest.h>
#include "test_config.hpp"
#include <lloyal/chat_in.hpp>
#include <lloyal/chat_out.hpp>
#include <lloyal/model_registry.hpp>
#include <nlohmann/json.hpp>
#include <llama/llama.h>
#include <cstdlib>
#include <string>

using json = nlohmann::ordered_json;

static const char* MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

#define REQUIRE_MODEL() \
  do { \
    if (!MODEL_PATH) { \
      MESSAGE("SKIP: LLAMA_TEST_MODEL not set"); \
      return; \
    } \
  } while (0)

// ===== BASIC PARSING TESTS =====

TEST_CASE("ChatOut Integration: parse plain text output") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Get format from chat_in for this model
  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  }).dump();
  auto fmt = lloyal::chat_in::format(model.get(), inputs);

  // Parse plain text with the detected format
  auto result = lloyal::chat_out::parse(
    "Hello, I'm an AI assistant!",
    fmt.format,
    fmt.reasoning_format
  );

  CHECK(!result.content.empty());
  CHECK(result.content.find("Hello") != std::string::npos);
  MESSAGE("Parsed content: \"" << result.content << "\"");
}

TEST_CASE("ChatOut Integration: parse with format from chat_in roundtrip") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Format a message with tools to get a non-trivial format
  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What time is it?"}}
  }).dump();
  inputs.tools_json = json::array({
    {{"type", "function"}, {"function", {
      {"name", "get_time"},
      {"description", "Get current time"},
      {"parameters", {{"type", "object"}}}
    }}}
  }).dump();

  auto fmt = lloyal::chat_in::format(model.get(), inputs);
  MESSAGE("Format from chat_in: " << static_cast<int>(fmt.format));

  // Parse a simple response with the detected format (pass parser for PEG models)
  auto result = lloyal::chat_out::parse(
    "It is currently 3:00 PM.",
    fmt.format,
    fmt.reasoning_format,
    false,
    fmt.thinking_forced_open,
    fmt.parser
  );

  CHECK(!result.content.empty());
  MESSAGE("Parsed content: \"" << result.content << "\"");
  MESSAGE("Tool calls: " << result.tool_calls.size());
}

// ===== MODEL-BASED AUTO-DETECT TESTS =====

TEST_CASE("ChatOut Integration: parse with model auto-detect") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto result = lloyal::chat_out::parse(
    model.get(),
    "This is a test response."
  );

  CHECK(!result.content.empty());
  CHECK(result.content.find("test response") != std::string::npos);
}

TEST_CASE("ChatOut Integration: parse partial output with model") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto result = lloyal::chat_out::parse(
    model.get(),
    "Partial respon",
    true  // is_partial
  );

  CHECK(!result.content.empty());
}

// ===== RESULT STRUCTURE TESTS =====

TEST_CASE("ChatOut Integration: parse returns empty tool_calls for plain text") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto result = lloyal::chat_out::parse(
    model.get(),
    "Just a plain text response with no tool calls."
  );

  CHECK(result.tool_calls.empty());
  CHECK(!result.content.empty());
}

TEST_CASE("ChatOut Integration: parse empty string") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto result = lloyal::chat_out::parse(
    model.get(),
    ""
  );

  CHECK(result.tool_calls.empty());
}
