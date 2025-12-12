/**
 * Integration tests for chat template round-trip pattern with real models
 *
 * Tests the full round-trip pattern with actual GGUF models:
 * 1. Load model and query GGUF metadata (add_bos, add_eos flags)
 * 2. Format template with metadata-aware conditional stripping
 * 3. Tokenize with metadata-aware BOS/EOS addition
 * 4. Verify tokens match model expectations
 *
 * Requires: LLAMA_TEST_MODEL and LLAMA_TEST_TINYLLAMA environment variables
 */

#include <doctest/doctest.h>
#include "test_config.hpp"
#include <lloyal/helpers.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/nlohmann/json.hpp>
#include <llama/llama.h>
#include <cstdlib>
#include <string>

using json = nlohmann::ordered_json;

// ===== TEST HELPERS =====

static const char* MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");
static const char* TINYLLAMA_PATH = std::getenv("LLAMA_TEST_TINYLLAMA");

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

#define REQUIRE_TINYLLAMA() \
  do { \
    if (!TINYLLAMA_PATH) { \
      MESSAGE("SKIP: LLAMA_TEST_TINYLLAMA not set"); \
      return; \
    } \
  } while (0)

// ===== METADATA VERIFICATION TESTS =====

TEST_CASE("ChatTemplate Integration: verify TinyLlama metadata flags") {
  REQUIRE_TINYLLAMA();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(TINYLLAMA_PATH, params);
  REQUIRE(model != nullptr);

  const llama_vocab* vocab = llama_model_get_vocab(model);
  REQUIRE(vocab != nullptr);

  // TinyLlama should have add_bos=false, add_eos=false
  bool add_bos = llama_vocab_get_add_bos(vocab);
  bool add_eos = llama_vocab_get_add_eos(vocab);

  CHECK(add_bos == false);
  CHECK(add_eos == false);

  llama_model_free(model);
}

TEST_CASE("ChatTemplate Integration: extract template from TinyLlama") {
  REQUIRE_TINYLLAMA();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(TINYLLAMA_PATH, params);
  REQUIRE(model != nullptr);

  const char* template_str = llama_model_chat_template(model, nullptr);
  REQUIRE(template_str != nullptr);
  REQUIRE(strlen(template_str) > 0);

  // TinyLlama uses Zephyr template format
  std::string tmpl(template_str);
  CHECK(tmpl.find("<|user|>") != std::string::npos);
  CHECK(tmpl.find("<|assistant|>") != std::string::npos);

  llama_model_free(model);
}

// ===== ROUND-TRIP PATTERN TESTS =====

TEST_CASE("ChatTemplate Integration: format with TinyLlama template") {
  REQUIRE_TINYLLAMA();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(TINYLLAMA_PATH, params);
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello, how are you?"}}
  });

  auto result = lloyal::format_chat_template_complete(model, messages.dump());

  // Should have formatted prompt
  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Hello, how are you?") != std::string::npos);

  // Should include template structure
  CHECK(result.prompt.find("<|user|>") != std::string::npos);
  CHECK(result.prompt.find("<|assistant|>") != std::string::npos);

  llama_model_free(model);
}

TEST_CASE("ChatTemplate Integration: multi-turn conversation with TinyLlama") {
  REQUIRE_TINYLLAMA();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(TINYLLAMA_PATH, params);
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "What is 2+2?"}},
    {{"role", "assistant"}, {"content", "4"}},
    {{"role", "user"}, {"content", "What is 3+3?"}}
  });

  auto result = lloyal::format_chat_template_complete(model, messages.dump());

  CHECK(!result.prompt.empty());
  // All messages should be present
  CHECK(result.prompt.find("What is 2+2?") != std::string::npos);
  CHECK(result.prompt.find("4") != std::string::npos);
  CHECK(result.prompt.find("What is 3+3?") != std::string::npos);

  llama_model_free(model);
}

TEST_CASE("ChatTemplate Integration: tokenization round-trip with metadata") {
  REQUIRE_TINYLLAMA();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(TINYLLAMA_PATH, params);
  REQUIRE(model != nullptr);

  const llama_vocab* vocab = llama_model_get_vocab(model);
  REQUIRE(vocab != nullptr);

  // Format a simple message
  json messages = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  });

  auto result = lloyal::format_chat_template_complete(model, messages.dump());
  REQUIRE(!result.prompt.empty());

  // Query metadata
  bool add_bos = llama_vocab_get_add_bos(vocab);

  // Tokenize with metadata-aware handling
  auto tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, add_bos, true);

  CHECK(!tokens.empty());

  // With TinyLlama (add_bos=false), the first token should NOT be BOS
  if (!add_bos) {
    llama_token bos = llama_vocab_bos(vocab);
    CHECK(tokens[0] != bos);
  }

  llama_model_free(model);
}

TEST_CASE("ChatTemplate Integration: stop tokens extraction") {
  REQUIRE_TINYLLAMA();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(TINYLLAMA_PATH, params);
  REQUIRE(model != nullptr);

  const char* template_str = llama_model_chat_template(model, nullptr);
  REQUIRE(template_str != nullptr);

  auto stops = lloyal::extract_template_stop_tokens(model, template_str);

  // Should have extracted some stop tokens
  CHECK(!stops.empty());

  llama_model_free(model);
}

// ===== TEMPLATE OVERRIDE TESTS =====

TEST_CASE("ChatTemplate Integration: custom template override") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(MODEL_PATH, params);
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

  auto result = lloyal::format_chat_template_complete(model, messages.dump(), override_template);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("<|im_start|>") != std::string::npos);
  CHECK(result.prompt.find("Test message") != std::string::npos);
  CHECK(result.prompt.find("<|im_end|>") != std::string::npos);

  llama_model_free(model);
}

// ===== EDGE CASE TESTS =====

TEST_CASE("ChatTemplate Integration: long conversation (50 turns)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(MODEL_PATH, params);
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

  auto result = lloyal::format_chat_template_complete(model, messages.dump());

  CHECK(!result.prompt.empty());
  // Check first and last messages are present
  CHECK(result.prompt.find("Message 0") != std::string::npos);
  CHECK(result.prompt.find("Response 99") != std::string::npos);

  llama_model_free(model);
}

TEST_CASE("ChatTemplate Integration: very long message content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  // 10KB message
  std::string long_content(10000, 'x');

  json messages = json::array({
    {{"role", "user"}, {"content", long_content}}
  });

  auto result = lloyal::format_chat_template_complete(model, messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find(long_content) != std::string::npos);

  llama_model_free(model);
}

TEST_CASE("ChatTemplate Integration: special characters in content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "Quote: \"Hello\"\nNewline\tTab\rCarriage"}}
  });

  auto result = lloyal::format_chat_template_complete(model, messages.dump());

  CHECK(!result.prompt.empty());

  llama_model_free(model);
}

TEST_CASE("ChatTemplate Integration: unicode and emoji content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto params = llama_model_default_params();
  params.n_gpu_layers = TestConfig::n_gpu_layers();

  llama_model *model = llama_model_load_from_file(MODEL_PATH, params);
  REQUIRE(model != nullptr);

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello 世界 🌍 Привет مرحبا"}}
  });

  auto result = lloyal::format_chat_template_complete(model, messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("世界") != std::string::npos);
  CHECK(result.prompt.find("🌍") != std::string::npos);

  llama_model_free(model);
}

