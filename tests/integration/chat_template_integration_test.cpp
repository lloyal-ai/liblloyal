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
#include <lloyal/chat_template.hpp>
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

  // Matches documented example in get_turn_separator() @code block
  // This verifies the warm multi-turn continuation pattern works

  // Step 1: Get turn separator
  auto separator = lloyal::chat_template::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Step 2: Format a "delta" (new user message)
  const auto* vocab = llama_model_get_vocab(model.get());
  std::string delta_prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
  auto delta_tokens = lloyal::tokenizer::tokenize(vocab, delta_prompt, false, true);
                                      // add_bos=false: continuing, not fresh start
  REQUIRE(!delta_tokens.empty());

  // Step 3: Prepend separator to delta for exact match with cold path
  std::vector<llama_token> prefill_tokens;
  prefill_tokens.insert(prefill_tokens.end(), separator.begin(), separator.end());
  prefill_tokens.insert(prefill_tokens.end(), delta_tokens.begin(), delta_tokens.end());

  // Verify the pattern produces valid tokens
  CHECK(prefill_tokens.size() == separator.size() + delta_tokens.size());
  CHECK(prefill_tokens.size() > 0);

  // First tokens should be the separator
  for (size_t i = 0; i < separator.size(); i++) {
    CHECK(prefill_tokens[i] == separator[i]);
  }

  MESSAGE("Warm continuation pattern: separator(" << separator.size()
          << ") + delta(" << delta_tokens.size()
          << ") = " << prefill_tokens.size() << " tokens");
}
