/**
 * Integration tests for chat template round-trip pattern with real models
 *
 * Tests the full round-trip pattern with actual GGUF models:
 * 1. Load model and query GGUF metadata (add_bos, add_eos flags)
 * 2. Format template with metadata-aware conditional stripping
 * 3. Tokenize with metadata-aware BOS/EOS addition
 * 4. Verify tokens match model expectations
 *
 * Requires: LLAMA_TEST_MODEL environment variable
 */

#include <doctest/doctest.h>
#include <lloyal/helpers.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/nlohmann/json.hpp>
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

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
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

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  const char* template_str = llama_model_chat_template(model.get(), nullptr);
  REQUIRE(template_str != nullptr);
  REQUIRE(strlen(template_str) > 0);

  // Template should contain Jinja2 syntax (model-agnostic)
  std::string tmpl(template_str);
  CHECK(tmpl.find("{%") != std::string::npos);
  CHECK(tmpl.find("message") != std::string::npos);
}

// ===== ROUND-TRIP PATTERN TESTS =====

TEST_CASE("ChatTemplate Integration: format with model template") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello, how are you?"}}
  });

  auto result = lloyal::format_chat_template_complete(model.get(), messages.dump());

  // Should have formatted prompt containing the message content
  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Hello, how are you?") != std::string::npos);

  // Template should have added structure (prompt longer than just the content)
  CHECK(result.prompt.size() > strlen("Hello, how are you?"));
}

TEST_CASE("ChatTemplate Integration: multi-turn conversation") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "What is 2+2?"}},
    {{"role", "assistant"}, {"content", "4"}},
    {{"role", "user"}, {"content", "What is 3+3?"}}
  });

  auto result = lloyal::format_chat_template_complete(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
  // All messages should be present
  CHECK(result.prompt.find("What is 2+2?") != std::string::npos);
  CHECK(result.prompt.find("4") != std::string::npos);
  CHECK(result.prompt.find("What is 3+3?") != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: tokenization round-trip with metadata") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Format a simple message
  json messages = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  });

  auto result = lloyal::format_chat_template_complete(model.get(), messages.dump());
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
  // Note: when add_bos=false we don't assert tokens[0] != bos because
  // the template itself may produce text that tokenizes to the BOS token ID
}

TEST_CASE("ChatTemplate Integration: stop tokens extraction") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  const char* template_str = llama_model_chat_template(model.get(), nullptr);
  REQUIRE(template_str != nullptr);

  auto stops = lloyal::extract_template_stop_tokens(model.get(), template_str);

  // Should have extracted some stop tokens
  CHECK(!stops.empty());
}

// ===== TEMPLATE OVERRIDE TESTS =====

TEST_CASE("ChatTemplate Integration: custom template override") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
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

  auto result = lloyal::format_chat_template_complete(model.get(), messages.dump(), override_template);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("<|im_start|>") != std::string::npos);
  CHECK(result.prompt.find("Test message") != std::string::npos);
  CHECK(result.prompt.find("<|im_end|>") != std::string::npos);
}

// ===== EDGE CASE TESTS =====

TEST_CASE("ChatTemplate Integration: long conversation (50 turns)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
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

  auto result = lloyal::format_chat_template_complete(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
  // Check first and last messages are present
  CHECK(result.prompt.find("Message 0") != std::string::npos);
  CHECK(result.prompt.find("Response 99") != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: very long message content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  // 10KB message
  std::string long_content(10000, 'x');

  json messages = json::array({
    {{"role", "user"}, {"content", long_content}}
  });

  auto result = lloyal::format_chat_template_complete(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find(long_content) != std::string::npos);
}

TEST_CASE("ChatTemplate Integration: special characters in content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "Quote: \"Hello\"\nNewline\tTab\rCarriage"}}
  });

  auto result = lloyal::format_chat_template_complete(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
}

TEST_CASE("ChatTemplate Integration: unicode and emoji content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;

  auto model = lloyal::ModelRegistry::acquire(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello 世界 🌍 Привет مرحبا"}}
  });

  auto result = lloyal::format_chat_template_complete(model.get(), messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("世界") != std::string::npos);
  CHECK(result.prompt.find("🌍") != std::string::npos);
}

