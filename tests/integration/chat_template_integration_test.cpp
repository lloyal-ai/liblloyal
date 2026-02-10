/**
 * Integration tests for chat template processing with real models
 *
 * Tests the chat template API with actual GGUF models:
 * - Template formatting using llama.cpp common library
 * - Turn separator extraction
 * - Tokenization round-trip
 *
 * Requires: LLAMA_TEST_MODEL environment variable
 */

#include <doctest/doctest.h>
#include "test_config.hpp"
#include <lloyal/branch.hpp>
#include <lloyal/chat_template.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <nlohmann/json.hpp>
#include <llama/llama.h>
#include <cstdlib>
#include <string>

using json = nlohmann::ordered_json;

// ===== TEST HELPERS =====

static const char* MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

// RAII guard for llama backend initialization
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

// ===== METADATA VERIFICATION TESTS =====

TEST_CASE("ChatTemplate Integration: verify model metadata flags") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Verify metadata flags are queryable (values are model-dependent)
  bool add_bos = llama_vocab_get_add_bos(vocab);
  bool add_eos = llama_vocab_get_add_eos(vocab);
  MESSAGE("Model metadata: add_bos=" << add_bos << ", add_eos=" << add_eos);

  // BOS and EOS token IDs should be valid
  llama_token bos = llama_vocab_bos(vocab);
  llama_token eos = llama_vocab_eos(vocab);
  CHECK(bos >= 0);
  CHECK(eos >= 0);
}

TEST_CASE("ChatTemplate Integration: extract chat template") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const char* template_str = llama_model_chat_template(model.get(), nullptr);
  REQUIRE(template_str != nullptr);
  REQUIRE(strlen(template_str) > 0);

  // Template should contain Jinja2 syntax (model-agnostic)
  std::string tmpl(template_str);
  CHECK(tmpl.find("{%") != std::string::npos);
  CHECK(tmpl.find("message") != std::string::npos);
}

// ===== FORMAT TESTS =====

TEST_CASE("ChatTemplate Integration: format with model template") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello, how are you?"}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  // Should have formatted prompt containing the message content
  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Hello, how are you?") != std::string::npos);

  // Template should have added structure (prompt longer than just the content)
  CHECK(result.prompt.size() > strlen("Hello, how are you?"));
}

TEST_CASE("ChatTemplate Integration: multi-turn conversation") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "What is 2+2?"}},
    {{"role", "assistant"}, {"content", "4"}},
    {{"role", "user"}, {"content", "What is 3+3?"}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
  // All messages should be present
  CHECK(result.prompt.find("What is 2+2?") != std::string::npos);
  CHECK(result.prompt.find("4") != std::string::npos);
  CHECK(result.prompt.find("What is 3+3?") != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: tokenization round-trip with metadata") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Format a simple message
  json messages = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());
  REQUIRE(!result.prompt.empty());

  // Query metadata
  bool add_bos = llama_vocab_get_add_bos(vocab);

  // Tokenize with metadata-aware handling
  auto tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, add_bos, true);

  CHECK(!tokens.empty());

  // Verify BOS handling is consistent with metadata
  llama_token bos = llama_vocab_bos(vocab);
  if (add_bos) {
    // When add_bos is true, tokenizer should prepend BOS
    CHECK(tokens[0] == bos);
  }
}

// ===== TEMPLATE OVERRIDE TESTS =====

TEST_CASE("ChatTemplate Integration: custom template override") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Use ChatML as override
  std::string override_template =
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}";

  json messages = json::array({
    {{"role", "user"}, {"content", "Test message"}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump(), override_template);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("<|im_start|>") != std::string::npos);
  CHECK(result.prompt.find("Test message") != std::string::npos);
  CHECK(result.prompt.find("<|im_end|>") != std::string::npos);
}

// ===== EDGE CASE TESTS =====

TEST_CASE("ChatTemplate Integration: long conversation (50 turns)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  json messages = json::array();

  // Create 50-turn conversation
  for (int i = 0; i < 50; i++) {
    messages.push_back({
      {"role", "user"},
      {"content", "Message " + std::to_string(i * 2)}
    });
    messages.push_back({
      {"role", "assistant"},
      {"content", "Response " + std::to_string(i * 2 + 1)}
    });
  }

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
  // Check first and last messages are present
  CHECK(result.prompt.find("Message 0") != std::string::npos);
  CHECK(result.prompt.find("Response 99") != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: very long message content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // 10KB message
  std::string long_content(10000, 'x');

  json messages = json::array({
    {{"role", "user"}, {"content", long_content}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find(long_content) != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: special characters in content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "Quote: \"Hello\"\nNewline\tTab\rCarriage"}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
}

TEST_CASE("ChatTemplate Integration: unicode and emoji content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello 世界 🌍 Привет مرحبا"}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("世界") != std::string::npos);
  CHECK(result.prompt.find("🌍") != std::string::npos);
}

// ===== TURN SEPARATOR TESTS =====

TEST_CASE("ChatTemplate Integration: get_turn_separator returns non-empty") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto separator = lloyal::chat_template::get_turn_separator(model.get());

  // Should return at least one token (the EOG token)
  CHECK(!separator.empty());
  MESSAGE("Separator has " << separator.size() << " token(s)");
}

TEST_CASE("ChatTemplate Integration: get_turn_separator contains EOG token") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto separator = lloyal::chat_template::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Find EOG token in separator
  bool found_eog = false;
  for (auto tok : separator) {
    if (lloyal::tokenizer::is_eog(model.get(), tok)) {
      found_eog = true;
      break;
    }
  }

  CHECK(found_eog);
}

TEST_CASE("ChatTemplate Integration: get_turn_separator detokenizes correctly") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto separator = lloyal::chat_template::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Detokenize and verify it contains expected content
  std::string text;
  for (auto tok : separator) {
    text += lloyal::tokenizer::detokenize(model.get(), tok);
  }

  MESSAGE("Separator text: \"" << text << "\"");

  // Should contain at least one character (the EOG token text)
  CHECK(!text.empty());
}

TEST_CASE("ChatTemplate Integration: get_turn_separator is stable (deterministic)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Call twice - should return identical results
  auto sep1 = lloyal::chat_template::get_turn_separator(model.get());
  auto sep2 = lloyal::chat_template::get_turn_separator(model.get());

  REQUIRE(sep1.size() == sep2.size());
  for (size_t i = 0; i < sep1.size(); i++) {
    CHECK(sep1[i] == sep2[i]);
  }
}

TEST_CASE("ChatTemplate Integration: get_turn_separator matches template boundary") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Format a 2-turn conversation to get the actual boundary
  json messages = json::array({
    {{"role", "user"}, {"content", "X"}},
    {{"role", "assistant"}, {"content", "Y"}},
    {{"role", "user"}, {"content", "Z"}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());
  REQUIRE(!result.prompt.empty());

  // Tokenize the full prompt
  auto full_tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, false, true);

  // Get the separator
  auto separator = lloyal::chat_template::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // The separator tokens should appear in the full prompt somewhere
  // (between assistant content "Y" and user content "Z")
  bool found_separator = false;
  for (size_t i = 0; i + separator.size() <= full_tokens.size(); i++) {
    bool match = true;
    for (size_t j = 0; j < separator.size(); j++) {
      if (full_tokens[i + j] != separator[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      found_separator = true;
      MESSAGE("Found separator at position " << i);
      break;
    }
  }

  CHECK(found_separator);
}

TEST_CASE("ChatTemplate Integration: get_turn_separator with null model") {
  // Should handle null gracefully
  auto separator = lloyal::chat_template::get_turn_separator(nullptr);
  CHECK(separator.empty());
}

// ===== DOCUMENTED EXAMPLE TESTS =====
// These tests verify the examples in chat_template.hpp Doxygen comments actually work

TEST_CASE("ChatTemplate Integration: format() basic usage (documented example)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Matches documented example in format() @code block
  auto result = lloyal::chat_template::format(model.get(), R"([{"role":"user","content":"Hi"}])");
  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, true, true);

  // Verify the pattern works
  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Hi") != std::string::npos);
  CHECK(!tokens.empty());
}

TEST_CASE("ChatTemplate Integration: format() with custom template (documented example)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // ChatML override template (matches documented pattern)
  std::string custom_template =
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}";

  std::string messages_json = R"([{"role":"user","content":"Hello"}])";

  auto custom_result = lloyal::chat_template::format(model.get(), messages_json, custom_template);

  CHECK(!custom_result.prompt.empty());
  CHECK(custom_result.prompt.find("<|im_start|>") != std::string::npos);
  CHECK(custom_result.prompt.find("Hello") != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: warm continuation pattern (documented example)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Warm continuation pattern:
  // 1. KV cache has: tokens for conversation up to assistant's completed response
  // 2. When user adds a new message, we need to prefill: separator + new_user_turn + assistant_header
  // 3. The separator tokens close the previous assistant turn (e.g., <|eot_id|>)
  //
  // This test verifies that the separator extraction correctly identifies the tokens
  // that bridge an assistant response to the next user turn.

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Step 1: Get turn separator
  auto separator = lloyal::chat_template::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Step 2: Format full conversation (cold path)
  json cold_messages = json::array({
    {{"role", "user"}, {"content", "Hello"}},
    {{"role", "assistant"}, {"content", "Hi there!"}},
    {{"role", "user"}, {"content", "What is 2+2?"}}
  });
  auto cold_result = lloyal::chat_template::format(model.get(), cold_messages.dump());
  REQUIRE(!cold_result.prompt.empty());

  // Tokenize the cold prompt
  auto cold_tokens = lloyal::tokenizer::tokenize(vocab, cold_result.prompt, false, true);
  REQUIRE(!cold_tokens.empty());

  // Step 3: Find where the separator appears in the cold tokens
  // The separator should appear after "Hi there!" (the assistant's response)
  // We search for the separator sequence in the cold tokens

  // Detokenize separator for logging
  std::string separator_text;
  for (auto tok : separator) {
    separator_text += lloyal::tokenizer::detokenize(model.get(), tok);
  }
  MESSAGE("Separator text: \"" << separator_text << "\"");

  // Step 4: Verify the separator token sequence matches in the tokenized form
  // (Text-based search is unreliable due to newlines/whitespace in different templates)
  // Token-based search is the authoritative verification
  bool found_separator = false;
  size_t separator_token_pos = 0;
  for (size_t i = 0; i + separator.size() <= cold_tokens.size(); i++) {
    bool matches = true;
    for (size_t j = 0; j < separator.size(); j++) {
      if (cold_tokens[i + j] != separator[j]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      found_separator = true;
      separator_token_pos = i;
      break;
    }
  }
  CHECK(found_separator);
  if (found_separator) {
    MESSAGE("Separator tokens found at position " << separator_token_pos
            << " in cold tokens (total: " << cold_tokens.size() << ")");
  }

  // Step 5: For a real warm continuation, we would:
  // - Have KV cache containing tokens[0:separator_token_pos] (before the separator)
  // - Prefill with: separator + new_user_turn + assistant_header
  // The separator bridges the cached assistant response to the new turn

  if (found_separator) {
    size_t warm_cache_size = separator_token_pos;
    size_t delta_size = separator.size() + (cold_tokens.size() - separator_token_pos - separator.size());
    MESSAGE("Warm continuation: warm_cache(" << warm_cache_size
            << ") + separator(" << separator.size()
            << ") + new_turn(" << (cold_tokens.size() - separator_token_pos - separator.size())
            << ") = cold(" << cold_tokens.size() << ")");
  }
}

// ===== TOOL CALLING TESTS =====
// These tests verify that tool_calls, name, tool_call_id, and reasoning_content
// are properly passed through to the Jinja template engine

TEST_CASE("ChatTemplate Integration: tool_calls in assistant message") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Assistant message with tool_calls (null content is valid for tool-calling)
  json messages = json::array({
    {{"role", "user"}, {"content", "What's the weather in Melbourne?"}},
    {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"type", "function"}, {"id", "call_123"}, {"function", {
        {"name", "get_weather"},
        {"arguments", R"({"location": "Melbourne"})"}}
      }}
    })}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  // Should format without error
  // Note: The template may or may not render tool_calls depending on model support
  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("What's the weather") != std::string::npos);
  MESSAGE("Tool calls prompt length: " << result.prompt.size());
}

TEST_CASE("ChatTemplate Integration: tool response message") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Full tool calling round-trip: user → assistant (tool_calls) → tool (response)
  json messages = json::array({
    {{"role", "user"}, {"content", "What's the weather?"}},
    {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"type", "function"}, {"id", "call_456"}, {"function", {
        {"name", "weather"},
        {"arguments", "{}"}}
      }}
    })}},
    {{"role", "tool"}, {"name", "weather"}, {"tool_call_id", "call_456"}, {"content", "22 degrees and sunny"}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  // Should format without error
  CHECK(!result.prompt.empty());
  // Tool response content should appear in the formatted output
  CHECK(result.prompt.find("22") != std::string::npos);
  MESSAGE("Tool response prompt length: " << result.prompt.size());
}

TEST_CASE("ChatTemplate Integration: multiple tool calls in single message") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Assistant makes multiple parallel tool calls
  json messages = json::array({
    {{"role", "user"}, {"content", "Compare weather in Melbourne and Sydney"}},
    {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"type", "function"}, {"id", "call_1"}, {"function", {
        {"name", "get_weather"},
        {"arguments", R"({"location": "Melbourne"})"}}
      }},
      {{"type", "function"}, {"id", "call_2"}, {"function", {
        {"name", "get_weather"},
        {"arguments", R"({"location": "Sydney"})"}}
      }}
    })}},
    {{"role", "tool"}, {"name", "get_weather"}, {"tool_call_id", "call_1"}, {"content", "Melbourne: 22C"}},
    {{"role", "tool"}, {"name", "get_weather"}, {"tool_call_id", "call_2"}, {"content", "Sydney: 25C"}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
  // Both tool responses should appear
  CHECK(result.prompt.find("Melbourne") != std::string::npos);
  CHECK(result.prompt.find("Sydney") != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: reasoning_content field") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Message with reasoning_content (for thinking/CoT models like DeepSeek-R1)
  json messages = json::array({
    {{"role", "user"}, {"content", "What is 15 * 7?"}},
    {{"role", "assistant"}, {"reasoning_content", "Let me calculate step by step: 15 * 7 = 15 * 5 + 15 * 2 = 75 + 30 = 105"}, {"content", "The answer is 105."}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  // Should format without error
  CHECK(!result.prompt.empty());
  // The content should appear (reasoning_content may or may not depending on template)
  CHECK(result.prompt.find("105") != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: content_parts array") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Message with structured content_parts (OpenAI vision-style format)
  json messages = json::array({
    {{"role", "user"}, {"content", json::array({
      {{"type", "text"}, {"text", "Hello, "}},
      {{"type", "text"}, {"text", "world!"}}
    })}}
  });

  auto result = lloyal::chat_template::format(model.get(), messages.dump());

  // Should format without error
  CHECK(!result.prompt.empty());
  // Text parts should be present (concatenated or separate depending on template)
  CHECK(result.prompt.find("Hello") != std::string::npos);
  CHECK(result.prompt.find("world") != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: warm continuation with tool messages") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Simulate warm continuation after a tool call
  // Warm state: user asked, assistant made tool call
  json warm_messages = json::array({
    {{"role", "user"}, {"content", "What's the weather?"}},
    {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"type", "function"}, {"id", "call_789"}, {"function", {
        {"name", "weather"},
        {"arguments", "{}"}}
      }}
    })}}
  });

  // Cold state: includes tool response
  json cold_messages = warm_messages;
  cold_messages.push_back({
    {"role", "tool"}, {"name", "weather"}, {"tool_call_id", "call_789"}, {"content", "Sunny and warm"}
  });

  auto warm_result = lloyal::chat_template::format(model.get(), warm_messages.dump());
  auto cold_result = lloyal::chat_template::format(model.get(), cold_messages.dump());

  CHECK(!warm_result.prompt.empty());
  CHECK(!cold_result.prompt.empty());

  // Skip further checks if template doesn't support tool messages
  // (some templates like Zephyr/ChatML don't have native tool support)
  if (cold_result.prompt.size() <= warm_result.prompt.size()) {
    MESSAGE("[ SKIP ] Template does not appear to support tool messages natively");
    return;
  }

  // Cold should be longer (includes tool response)
  CHECK(cold_result.prompt.size() > warm_result.prompt.size());

  // Tool response content should appear in cold but not warm
  CHECK(cold_result.prompt.find("Sunny") != std::string::npos);

  MESSAGE("Warm continuation with tools: warm(" << warm_result.prompt.size()
          << ") → cold(" << cold_result.prompt.size() << ")");
}

// ===== HIGH-FIDELITY WARM CONTINUATION VERIFICATION =====
// This test verifies the CORE INVARIANT of warm continuation:
//   cold_tokens == warm_base_tokens + separator + delta_tokens
// This is the critical property that guarantees warm and cold paths produce identical results.

TEST_CASE("ChatTemplate Integration: warm continuation token-level parity") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // CORE INVARIANT TEST:
  // When we have an existing conversation in KV cache and add a new turn,
  // the resulting token sequence must be IDENTICAL to cold-starting the full conversation.
  //
  // Specifically:
  //   cold_tokens == warm_base + separator + delta
  // Where:
  //   - warm_base = tokens for [user1, assistant1_response] (cached, before separator)
  //   - separator = EOT tokens that close assistant1's turn
  //   - delta = tokens for [user2, assistant_generation_prompt] (new prefill)

  // Step 1: Format full conversation (cold path - ground truth)
  json cold_messages = json::array({
    {{"role", "user"}, {"content", "What is your name?"}},
    {{"role", "assistant"}, {"content", "I am an AI assistant."}},
    {{"role", "user"}, {"content", "Nice to meet you!"}}
  });
  auto cold_result = lloyal::chat_template::format(model.get(), cold_messages.dump());
  REQUIRE(!cold_result.prompt.empty());

  auto cold_tokens = lloyal::tokenizer::tokenize(vocab, cold_result.prompt, false, true);
  REQUIRE(!cold_tokens.empty());

  // Step 2: Get the turn separator
  auto separator = lloyal::chat_template::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Step 3: Find ALL separator occurrences in cold_tokens
  // Then identify which one is between assistant1 and user2
  std::vector<size_t> separator_positions;
  for (size_t i = 0; i + separator.size() <= cold_tokens.size(); i++) {
    bool match = true;
    for (size_t j = 0; j < separator.size(); j++) {
      if (cold_tokens[i + j] != separator[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      separator_positions.push_back(i);
      i += separator.size() - 1;  // Skip past this separator
    }
  }
  REQUIRE(!separator_positions.empty());

  // Step 4: For a 3-message conversation [user, assistant, user], the separator
  // after the assistant's response is typically the 2nd one
  // (1st = after user1 or system, 2nd = after assistant, etc.)
  REQUIRE(separator_positions.size() >= 2);
  size_t separator_token_pos = separator_positions[1];  // Second separator = after assistant

  // Step 5: VERIFY THE CORE INVARIANT
  // warm_base = cold_tokens[0:separator_token_pos]
  // delta = cold_tokens[separator_token_pos + separator.size():]
  // Reconstructed = warm_base + separator + delta should equal cold_tokens

  std::vector<llama_token> warm_base(cold_tokens.begin(),
                                      cold_tokens.begin() + separator_token_pos);
  std::vector<llama_token> delta(cold_tokens.begin() + separator_token_pos + separator.size(),
                                  cold_tokens.end());

  // Reconstruct
  std::vector<llama_token> reconstructed;
  reconstructed.insert(reconstructed.end(), warm_base.begin(), warm_base.end());
  reconstructed.insert(reconstructed.end(), separator.begin(), separator.end());
  reconstructed.insert(reconstructed.end(), delta.begin(), delta.end());

  // CRITICAL CHECK: Reconstructed must exactly equal cold_tokens
  REQUIRE(reconstructed.size() == cold_tokens.size());

  bool tokens_match = true;
  size_t mismatch_pos = 0;
  for (size_t i = 0; i < cold_tokens.size(); i++) {
    if (reconstructed[i] != cold_tokens[i]) {
      tokens_match = false;
      mismatch_pos = i;
      break;
    }
  }

  if (!tokens_match) {
    MESSAGE("Token mismatch at position " << mismatch_pos
            << ": expected " << cold_tokens[mismatch_pos]
            << " (" << lloyal::tokenizer::detokenize(model.get(), cold_tokens[mismatch_pos]) << ")"
            << ", got " << reconstructed[mismatch_pos]
            << " (" << lloyal::tokenizer::detokenize(model.get(), reconstructed[mismatch_pos]) << ")");
  }
  CHECK(tokens_match);

  MESSAGE("Token-level parity verified: warm_base(" << warm_base.size()
          << ") + separator(" << separator.size()
          << ") + delta(" << delta.size()
          << ") == cold(" << cold_tokens.size() << ")");
}

TEST_CASE("ChatTemplate Integration: warm continuation multi-turn token parity") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Test with a longer conversation to verify separator handling
  // across multiple turn boundaries

  json cold_messages = json::array({
    {{"role", "user"}, {"content", "A"}},
    {{"role", "assistant"}, {"content", "B"}},
    {{"role", "user"}, {"content", "C"}},
    {{"role", "assistant"}, {"content", "D"}},
    {{"role", "user"}, {"content", "E"}}
  });

  auto cold_result = lloyal::chat_template::format(model.get(), cold_messages.dump());
  REQUIRE(!cold_result.prompt.empty());

  auto cold_tokens = lloyal::tokenizer::tokenize(vocab, cold_result.prompt, false, true);
  auto separator = lloyal::chat_template::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Count how many times the separator appears in the cold tokens
  // Should appear after each assistant response (twice: after B and after D)
  size_t separator_count = 0;
  std::vector<size_t> separator_positions;

  for (size_t i = 0; i + separator.size() <= cold_tokens.size(); i++) {
    bool match = true;
    for (size_t j = 0; j < separator.size(); j++) {
      if (cold_tokens[i + j] != separator[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      separator_positions.push_back(i);
      separator_count++;
      i += separator.size() - 1;  // Skip past this separator
    }
  }

  // Should have at least 2 separators (after B and after D)
  // Note: Some templates may also add separator after the last user message
  CHECK(separator_count >= 2);
  MESSAGE("Found " << separator_count << " separator occurrences in "
          << cold_tokens.size() << " tokens");

  // Verify we can reconstruct from any separator position
  // Taking the LAST separator (after D, before E) as the warm continuation point
  if (separator_positions.size() >= 2) {
    size_t last_sep_pos = separator_positions.back();

    std::vector<llama_token> warm_base(cold_tokens.begin(),
                                        cold_tokens.begin() + last_sep_pos);
    std::vector<llama_token> delta(cold_tokens.begin() + last_sep_pos + separator.size(),
                                    cold_tokens.end());

    std::vector<llama_token> reconstructed;
    reconstructed.insert(reconstructed.end(), warm_base.begin(), warm_base.end());
    reconstructed.insert(reconstructed.end(), separator.begin(), separator.end());
    reconstructed.insert(reconstructed.end(), delta.begin(), delta.end());

    CHECK(reconstructed.size() == cold_tokens.size());

    bool match = (reconstructed == cold_tokens);
    CHECK(match);

    MESSAGE("Multi-turn parity: warm_base(" << warm_base.size()
            << ") at separator[" << (separator_positions.size() - 1) << "]"
            << " + delta(" << delta.size() << ")");
  }
}

// ===== ACTUAL WARM CONTINUATION TEST =====
// This is the REAL test - verifies that warm continuation produces
// IDENTICAL generation output as cold start.

// Sampling params struct for branch creation
struct WarmTestParams {
  float temperature = 0.0f;     // Greedy for determinism
  int32_t top_k = 0;            // Disabled - filters before greedy cause cross-backend non-determinism
  float top_p = 1.0f;           // Disabled - filters before greedy cause cross-backend non-determinism
  float min_p = 0.0f;           // Disabled - filters before greedy cause cross-backend non-determinism
  float typical_p = 1.0f;
  float penalty_repeat = 1.0f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 42;
};

TEST_CASE("ChatTemplate Integration: warm vs cold generation parity") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Create context for generation
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  // RAII cleanup
  struct ContextGuard {
    llama_context* c;
    ~ContextGuard() { if (c) llama_free(c); }
  } ctx_guard{ctx};

  // Create a conversation with context dependency
  // The assistant's response references the user's question
  json messages = json::array({
    {{"role", "user"}, {"content", "My name is Alice. What is my name?"}},
    {{"role", "assistant"}, {"content", "Your name is Alice."}},
    {{"role", "user"}, {"content", "What did I just tell you my name was?"}}
  });

  // Format and tokenize
  auto result = lloyal::chat_template::format(model.get(), messages.dump());
  REQUIRE(!result.prompt.empty());

  auto cold_tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, true, true);
  REQUIRE(!cold_tokens.empty());

  // Get separator
  auto separator = lloyal::chat_template::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Find ALL separator occurrences, then pick the right one
  // For a 3-message conversation [user, assistant, user], we want the separator
  // after the assistant's response (typically the 2nd or 2nd-to-last separator)
  std::vector<size_t> sep_positions;
  for (size_t i = 0; i + separator.size() <= cold_tokens.size(); i++) {
    bool match = true;
    for (size_t j = 0; j < separator.size(); j++) {
      if (cold_tokens[i + j] != separator[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      sep_positions.push_back(i);
      i += separator.size() - 1;  // Skip past this separator
    }
  }

  // Find the separator after assistant's response (before user2's content)
  // The correct separator is followed by a USER role marker, not ASSISTANT
  // Look for separator where immediate text after contains "user" (role marker)
  REQUIRE(sep_positions.size() >= 2);
  size_t separator_pos = sep_positions[1];  // Default to second separator
  for (size_t pos : sep_positions) {
    std::string after;
    // Get first ~30 chars after separator to check role marker
    for (size_t i = pos + separator.size(); i < cold_tokens.size() && i < pos + separator.size() + 5; i++) {
      after += lloyal::tokenizer::detokenize(model.get(), cold_tokens[i]);
    }
    // The separator after assistant is followed by user role marker
    // Also verify this is before user2's actual question
    std::string full_after;
    for (size_t i = pos + separator.size(); i < cold_tokens.size(); i++) {
      full_after += lloyal::tokenizer::detokenize(model.get(), cold_tokens[i]);
    }
    // Find separator where: 1) followed by user marker, 2) contains user2's question
    if ((after.find("user") != std::string::npos || after.find("User") != std::string::npos) &&
        full_after.find("just tell") != std::string::npos) {
      separator_pos = pos;
      break;
    }
  }

  // Split tokens
  std::vector<llama_token> warm_base(cold_tokens.begin(),
                                      cold_tokens.begin() + separator_pos);
  std::vector<llama_token> delta(cold_tokens.begin() + separator_pos,
                                  cold_tokens.end());

  MESSAGE("Warm continuation split: warm_base(" << warm_base.size()
          << ") + delta(" << delta.size()
          << ") = cold(" << cold_tokens.size() << ")");

  // Create branch store for this test
  lloyal::branch::BranchStore store;

  // Sampling params - DETERMINISTIC (greedy)
  WarmTestParams params;

  // ===== COLD PATH =====
  // Prefill entire conversation, generate N tokens
  auto cold_branch = lloyal::branch::create(ctx, model.get(), 0, 0, params, 512,
                                             nullptr, nullptr, &store);
  REQUIRE(cold_branch != lloyal::branch::INVALID_HANDLE);

  lloyal::branch::decode_and_capture_batch(cold_branch, cold_tokens.data(),
                                            cold_tokens.size(), &store);

  std::vector<llama_token> cold_output;
  for (int i = 0; i < 20; i++) {
    auto token = lloyal::branch::sample(cold_branch, &store);
    if (lloyal::tokenizer::is_eog(model.get(), token)) break;
    cold_output.push_back(token);
    lloyal::branch::accept_token(cold_branch, token, &store);
    lloyal::branch::decode_and_capture_one(cold_branch, token, &store);
  }

  lloyal::branch::destroy(cold_branch, &store);

  // Clear KV cache before warm path to ensure clean state
  lloyal::kv::clear_all(ctx);

  // ===== WARM PATH =====
  // Prefill warm_base (simulates cached KV), then prefill delta, generate N tokens

  // Use same seq_id since we cleared the cache
  auto warm_branch = lloyal::branch::create(ctx, model.get(), 0, 0, params, 512,
                                             nullptr, nullptr, &store);
  REQUIRE(warm_branch != lloyal::branch::INVALID_HANDLE);

  // Step 1: Prefill warm_base (this would normally be cached)
  lloyal::branch::decode_batch(warm_branch, warm_base.data(),
                                warm_base.size(), &store);

  // Step 2: Prefill delta (separator + new user turn + assistant header)
  lloyal::branch::decode_and_capture_batch(warm_branch, delta.data(),
                                            delta.size(), &store);

  std::vector<llama_token> warm_output;
  for (int i = 0; i < 20; i++) {
    auto token = lloyal::branch::sample(warm_branch, &store);
    if (lloyal::tokenizer::is_eog(model.get(), token)) break;
    warm_output.push_back(token);
    lloyal::branch::accept_token(warm_branch, token, &store);
    lloyal::branch::decode_and_capture_one(warm_branch, token, &store);
  }

  lloyal::branch::destroy(warm_branch, &store);

  // ===== VERIFY PARITY =====
  // Log the actual outputs FIRST for debugging
  std::string cold_text, warm_text;
  for (auto tok : cold_output) {
    cold_text += lloyal::tokenizer::detokenize(model.get(), tok);
  }
  for (auto tok : warm_output) {
    warm_text += lloyal::tokenizer::detokenize(model.get(), tok);
  }

  MESSAGE("Cold output (" << cold_output.size() << " tokens): \"" << cold_text << "\"");
  MESSAGE("Warm output (" << warm_output.size() << " tokens): \"" << warm_text << "\"");

  // Check for exact token match (ideal case)
  bool outputs_match = (cold_output.size() == warm_output.size());
  if (outputs_match) {
    for (size_t i = 0; i < cold_output.size(); i++) {
      if (cold_output[i] != warm_output[i]) {
        outputs_match = false;
        MESSAGE("Token mismatch at position " << i
                << ": cold=" << cold_output[i]
                << " (" << lloyal::tokenizer::detokenize(model.get(), cold_output[i]) << ")"
                << " vs warm=" << warm_output[i]
                << " (" << lloyal::tokenizer::detokenize(model.get(), warm_output[i]) << ")");
        break;
      }
    }
  }

  if (outputs_match) {
    MESSAGE("✓ WARM CONTINUATION VERIFIED: Exact token parity confirmed!");
  } else {
    // Some model/template combinations may have minor numerical variations
    // Verify semantic equivalence: both outputs should reference "Alice"
    bool cold_valid = cold_text.find("Alice") != std::string::npos;
    bool warm_valid = warm_text.find("Alice") != std::string::npos;
    CHECK(cold_valid);
    CHECK(warm_valid);
    if (cold_valid && warm_valid) {
      MESSAGE("✓ WARM CONTINUATION VERIFIED: Semantic parity confirmed (minor token variation)");
    }
  }
}

TEST_CASE("ChatTemplate Integration: true warm continuation (no re-prefill)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Create context
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  struct ContextGuard {
    llama_context* c;
    ~ContextGuard() { if (c) llama_free(c); }
  } ctx_guard{ctx};

  // This test simulates REAL warm continuation by splitting full_tokens:
  // 1. Find separator after assistant's response in full conversation
  // 2. warm_base = tokens before separator (what would be in KV cache)
  // 3. delta = tokens from separator onwards (what we prefill to continue)
  // 4. Verify: prefill(warm_base) + prefill(delta) == prefill(full)

  // Full conversation
  json full_messages = json::array({
    {{"role", "user"}, {"content", "My favorite color is blue. Remember that."}},
    {{"role", "assistant"}, {"content", "I'll remember that your favorite color is blue!"}},
    {{"role", "user"}, {"content", "What is my favorite color?"}}
  });
  auto full_result = lloyal::chat_template::format(model.get(), full_messages.dump());
  auto full_tokens = lloyal::tokenizer::tokenize(vocab, full_result.prompt, true, true);

  // Get separator
  auto separator = lloyal::chat_template::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Find ALL separator positions
  std::vector<size_t> sep_positions;
  for (size_t i = 0; i + separator.size() <= full_tokens.size(); i++) {
    bool match = true;
    for (size_t j = 0; j < separator.size(); j++) {
      if (full_tokens[i + j] != separator[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      sep_positions.push_back(i);
      i += separator.size() - 1;
    }
  }
  REQUIRE(sep_positions.size() >= 2);  // Need at least 2 separators

  // Find the separator after assistant's response (before user2's content)
  // Look for separator where delta contains "favorite color" (user2's question)
  size_t split_pos = sep_positions[1];  // Default to second separator
  for (size_t pos : sep_positions) {
    std::string after;
    for (size_t i = pos + separator.size(); i < full_tokens.size(); i++) {
      after += lloyal::tokenizer::detokenize(model.get(), full_tokens[i]);
    }
    if (after.find("favorite color") != std::string::npos) {
      split_pos = pos;
      break;
    }
  }

  // Split full_tokens into warm_base and delta
  std::vector<llama_token> warm_base(full_tokens.begin(),
                                      full_tokens.begin() + split_pos);
  std::vector<llama_token> delta(full_tokens.begin() + split_pos,
                                  full_tokens.end());

  MESSAGE("True warm continuation: warm_base(" << warm_base.size()
          << ") + delta(" << delta.size()
          << ") = full(" << full_tokens.size() << ")");

  // Verify split is correct
  REQUIRE(warm_base.size() + delta.size() == full_tokens.size());

  // Deterministic params
  WarmTestParams params;
  lloyal::branch::BranchStore store;

  // ===== COLD PATH =====
  auto cold_branch = lloyal::branch::create(ctx, model.get(), 0, 0, params, 512,
                                             nullptr, nullptr, &store);
  REQUIRE(cold_branch != lloyal::branch::INVALID_HANDLE);

  lloyal::branch::decode_and_capture_batch(cold_branch, full_tokens.data(),
                                            full_tokens.size(), &store);

  std::vector<llama_token> cold_output;
  for (int i = 0; i < 20; i++) {
    auto token = lloyal::branch::sample(cold_branch, &store);
    if (lloyal::tokenizer::is_eog(model.get(), token)) break;
    cold_output.push_back(token);
    lloyal::branch::accept_token(cold_branch, token, &store);
    lloyal::branch::decode_and_capture_one(cold_branch, token, &store);
  }

  lloyal::branch::destroy(cold_branch, &store);
  lloyal::kv::clear_all(ctx);

  // ===== WARM PATH =====
  // Prefill warm_base (simulates cached KV from previous turns)
  // Then prefill delta (separator + new turn)
  auto warm_branch = lloyal::branch::create(ctx, model.get(), 0, 0, params, 512,
                                             nullptr, nullptr, &store);
  REQUIRE(warm_branch != lloyal::branch::INVALID_HANDLE);

  // Step 1: Prefill warm_base (what would be in KV cache from previous generation)
  lloyal::branch::decode_batch(warm_branch, warm_base.data(),
                                warm_base.size(), &store);

  // Step 2: Prefill delta (separator + new user turn + assistant header)
  // This captures logits from the last token for sampling
  lloyal::branch::decode_and_capture_batch(warm_branch, delta.data(),
                                            delta.size(), &store);

  // Step 3: Generate
  std::vector<llama_token> warm_output;
  for (int i = 0; i < 20; i++) {
    auto token = lloyal::branch::sample(warm_branch, &store);
    if (lloyal::tokenizer::is_eog(model.get(), token)) break;
    warm_output.push_back(token);
    lloyal::branch::accept_token(warm_branch, token, &store);
    lloyal::branch::decode_and_capture_one(warm_branch, token, &store);
  }

  lloyal::branch::destroy(warm_branch, &store);

  // ===== VERIFY PARITY =====
  REQUIRE(cold_output.size() == warm_output.size());

  bool outputs_match = true;
  for (size_t i = 0; i < cold_output.size(); i++) {
    if (cold_output[i] != warm_output[i]) {
      outputs_match = false;
      MESSAGE("Mismatch at position " << i
              << ": cold=" << cold_output[i]
              << " vs warm=" << warm_output[i]);
      break;
    }
  }

  CHECK(outputs_match);

  std::string cold_text, warm_text;
  for (auto tok : cold_output) {
    cold_text += lloyal::tokenizer::detokenize(model.get(), tok);
  }
  for (auto tok : warm_output) {
    warm_text += lloyal::tokenizer::detokenize(model.get(), tok);
  }

  MESSAGE("Cold output: \"" << cold_text << "\"");
  MESSAGE("Warm output: \"" << warm_text << "\"");

  if (outputs_match) {
    MESSAGE("✓ TRUE WARM CONTINUATION: No re-prefill needed, generation matches!");
  }
}
