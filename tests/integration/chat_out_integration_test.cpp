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
#include <lloyal/branch.hpp>
#include <lloyal/chat_in.hpp>
#include <lloyal/chat_out.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/model_registry.hpp>
#include <nlohmann/json.hpp>
#include <llama/llama.h>
#include <cstdlib>
#include <optional>
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

// ===== GENERATION =====
//
// An integration test parses what the model actually writes: the prompt is prefilled and decoded through the
// framework's branch, greedily, so a run is repeatable.

struct GreedyParams {
  float temperature = 0.0f;
  int32_t top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float typical_p = 1.0f;
  float penalty_repeat = 1.0f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 42;
};

/** What the model writes for a formatted prompt: prefilled, then decoded greedily until EOG or `max_tokens`. */
static std::string generate(llama_model* model, const std::string& prompt, int max_tokens) {
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 4096;
  ctx_params.n_batch = 512;
  llama_context* ctx = llama_init_from_model(model, ctx_params);
  REQUIRE(ctx != nullptr);
  struct ContextGuard { llama_context* c; ~ContextGuard() { if (c) llama_free(c); } } ctx_guard{ctx};

  lloyal::branch::BranchStore store(4);
  store.init_tenancy(ctx);
  GreedyParams params;
  auto branch = lloyal::branch::create(ctx, model, store, 0, params, 512);
  REQUIRE(branch != lloyal::branch::INVALID_HANDLE);
  auto tokens = lloyal::tokenizer::tokenize(llama_model_get_vocab(model), prompt, true, true);
  lloyal::branch::prefill(branch, tokens.data(), tokens.size(), store);

  std::string text;
  for (int i = 0; i < max_tokens; i++) {
    auto token = lloyal::branch::sample(branch, store);
    if (lloyal::tokenizer::is_eog(model, token)) break;
    text += lloyal::tokenizer::detokenize(model, token);
    lloyal::branch::accept_token(branch, token, store);
    lloyal::branch::step(branch, token, store);
  }
  lloyal::branch::prune(branch, store);
  return text;
}

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

  // What the model writes for that prompt, parsed with the format, parser and generation prompt chat_in handed
  // back: whether it answers or calls the tool, the parse yields it, and no template marker is left in the reply.
  const std::string output = generate(model.get(), fmt.prompt, 1024);
  MESSAGE("generated: \"" << output << "\"");
  auto result = lloyal::chat_out::parse(
    output,
    fmt.format,
    fmt.reasoning_format,
    false,
    fmt.generation_prompt,
    fmt.parser
  );

  MESSAGE("Parsed content: \"" << result.content << "\"");
  MESSAGE("Tool calls: " << result.tool_calls.size());
  CHECK((!result.content.empty() || !result.tool_calls.empty()));
  if (fmt.supports_thinking) {
    CHECK(result.content.find(fmt.thinking_start_tag) == std::string::npos);
    CHECK(result.content.find(fmt.thinking_end_tag) == std::string::npos);
  }
}

// ===== THINKING MODELS =====
//
// A thinking template prefills the reasoning opener in its generation prompt, so what the model writes carries
// only the close. Formatted with chat_in's defaults, decoded for real, and parsed with the pairing chat_in hands
// back, the model's reasoning and its reply must come apart — the reply is what a caller shows, commits and
// keeps. Which tags those are is the TEMPLATE's to say (`supports_thinking`, `thinking_end_tag`), so these read
// as well for Magistral or gpt-oss as for Qwen. A template without thinking says so and the test states its
// skip; CI's SmolLM2 is one, so these run against a reasoning model (LLAMA_TEST_MODEL=Qwen3.5-4B) locally.

/** The format a caller gets by default (thinking is on by default), or nullopt for a template that declares no
 *  thinking at all. */
static std::optional<lloyal::chat_in::FormatResult> thinking_format(const llama_model* model, const std::string& user,
                                                                   const std::string& tools_json = "") {
  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({{{"role", "user"}, {"content", user}}}).dump();
  inputs.tools_json = tools_json;
  auto fmt = lloyal::chat_in::format(model, inputs);
  if (!fmt.supports_thinking || fmt.thinking_end_tag.empty()) return std::nullopt;
  return fmt;
}

TEST_CASE("ChatOut Integration: a thinking model's reply parses apart from its reasoning, with chat_in's defaults") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  auto fmt = thinking_format(model.get(), "What is 2 + 2? Reply with the number only.");
  if (!fmt) { MESSAGE("SKIP: this model's template declares no thinking"); return; }

  const std::string output = generate(model.get(), fmt->prompt, 1024);
  MESSAGE("generated: \"" << output << "\"");
  REQUIRE_MESSAGE(output.find(fmt->thinking_end_tag) != std::string::npos, "the model closed its reasoning within the budget");

  auto result = lloyal::chat_out::parse(output, fmt->format, fmt->reasoning_format, false, fmt->generation_prompt, fmt->parser);
  MESSAGE("content: \"" << result.content << "\"");
  CHECK(result.content.find(fmt->thinking_end_tag) == std::string::npos);
  CHECK(result.content.find(fmt->thinking_start_tag) == std::string::npos);
  CHECK(result.content.find('4') != std::string::npos);
  CHECK(!result.reasoning_content.empty());
}

TEST_CASE("ChatOut Integration: a thinking model's tool call parses, and its reasoning comes apart from it") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  const std::string tools = json::array({
    {{"type", "function"}, {"function", {
      {"name", "get_time"}, {"description", "Get the current time in a time zone"},
      {"parameters", {{"type", "object"}, {"properties", {{"zone", {{"type", "string"}}}}}, {"required", json::array({"zone"})}}}
    }}}
  }).dump();
  auto fmt = thinking_format(model.get(), "What time is it in UTC? Use the tool.", tools);
  if (!fmt) { MESSAGE("SKIP: this model's template declares no thinking"); return; }

  const std::string output = generate(model.get(), fmt->prompt, 1024);
  MESSAGE("generated: \"" << output << "\"");
  auto result = lloyal::chat_out::parse(output, fmt->format, fmt->reasoning_format, false, fmt->generation_prompt, fmt->parser);

  REQUIRE(result.tool_calls.size() == 1);
  CHECK(result.tool_calls[0].name == "get_time");
  CHECK(result.content.find(fmt->thinking_end_tag) == std::string::npos);
  CHECK(!result.reasoning_content.empty());
}

TEST_CASE("ChatOut Integration: how reasoning is parsed does not change the prompt a model is given") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "system"}, {"content", "You are terse."}},
    {{"role", "user"}, {"content", "What is 2 + 2?"}}
  }).dump();
  inputs.reasoning_format = "none";
  const auto none = lloyal::chat_in::format(model.get(), inputs);
  inputs.reasoning_format = "auto";
  const auto automatic = lloyal::chat_in::format(model.get(), inputs);
  CHECK(none.prompt == automatic.prompt);
  CHECK(none.generation_prompt == automatic.generation_prompt);
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
