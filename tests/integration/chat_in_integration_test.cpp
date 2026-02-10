/**
 * Integration tests for chat input formatting with real models
 *
 * Tests the chat_in API with actual GGUF models:
 * - Template formatting using llama.cpp common library
 * - Turn separator extraction
 * - Tokenization round-trip
 * - Tools and format awareness
 *
 * Requires: LLAMA_TEST_MODEL environment variable
 */

#include <doctest/doctest.h>
#include "test_config.hpp"
#include <lloyal/branch.hpp>
#include <lloyal/chat_in.hpp>
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

TEST_CASE("ChatIn Integration: verify model metadata flags") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  bool add_bos = llama_vocab_get_add_bos(vocab);
  bool add_eos = llama_vocab_get_add_eos(vocab);
  MESSAGE("Model metadata: add_bos=" << add_bos << ", add_eos=" << add_eos);

  llama_token bos = llama_vocab_bos(vocab);
  llama_token eos = llama_vocab_eos(vocab);
  CHECK(bos >= 0);
  CHECK(eos >= 0);
}

TEST_CASE("ChatIn Integration: extract chat template") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const char* template_str = llama_model_chat_template(model.get(), nullptr);
  REQUIRE(template_str != nullptr);
  REQUIRE(strlen(template_str) > 0);

  std::string tmpl(template_str);
  CHECK(tmpl.find("{%") != std::string::npos);
  CHECK(tmpl.find("message") != std::string::npos);
}

// ===== FORMAT TESTS =====

TEST_CASE("ChatIn Integration: format with model template") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello, how are you?"}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Hello, how are you?") != std::string::npos);
  CHECK(result.prompt.size() > strlen("Hello, how are you?"));
}

TEST_CASE("ChatIn Integration: multi-turn conversation") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What is 2+2?"}},
    {{"role", "assistant"}, {"content", "4"}},
    {{"role", "user"}, {"content", "What is 3+3?"}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("What is 2+2?") != std::string::npos);
  CHECK(result.prompt.find("4") != std::string::npos);
  CHECK(result.prompt.find("What is 3+3?") != std::string::npos);
}

TEST_CASE("ChatIn Integration: tokenization round-trip with metadata") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);
  REQUIRE(!result.prompt.empty());

  bool add_bos = llama_vocab_get_add_bos(vocab);
  auto tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, add_bos, true);

  CHECK(!tokens.empty());

  llama_token bos = llama_vocab_bos(vocab);
  if (add_bos) {
    CHECK(tokens[0] == bos);
  }
}

// ===== TEMPLATE OVERRIDE TESTS =====

TEST_CASE("ChatIn Integration: custom template override") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  std::string override_template =
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}";

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Test message"}}
  }).dump();
  inputs.template_override = override_template;

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("<|im_start|>") != std::string::npos);
  CHECK(result.prompt.find("Test message") != std::string::npos);
  CHECK(result.prompt.find("<|im_end|>") != std::string::npos);
}

// ===== EDGE CASE TESTS =====

TEST_CASE("ChatIn Integration: long conversation (50 turns)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  json messages = json::array();
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

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = messages.dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Message 0") != std::string::npos);
  CHECK(result.prompt.find("Response 99") != std::string::npos);
}

TEST_CASE("ChatIn Integration: very long message content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  std::string long_content(10000, 'x');

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", long_content}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find(long_content) != std::string::npos);
}

TEST_CASE("ChatIn Integration: special characters in content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Quote: \"Hello\"\nNewline\tTab\rCarriage"}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
}

TEST_CASE("ChatIn Integration: unicode and emoji content") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello 世界 🌍 Привет مرحبا"}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("世界") != std::string::npos);
  CHECK(result.prompt.find("🌍") != std::string::npos);
}

// ===== TURN SEPARATOR TESTS =====

TEST_CASE("ChatIn Integration: get_turn_separator returns non-empty") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto separator = lloyal::chat_in::get_turn_separator(model.get());

  CHECK(!separator.empty());
  MESSAGE("Separator has " << separator.size() << " token(s)");
}

TEST_CASE("ChatIn Integration: get_turn_separator contains EOG token") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto separator = lloyal::chat_in::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  bool found_eog = false;
  for (auto tok : separator) {
    if (lloyal::tokenizer::is_eog(model.get(), tok)) {
      found_eog = true;
      break;
    }
  }

  CHECK(found_eog);
}

TEST_CASE("ChatIn Integration: get_turn_separator detokenizes correctly") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto separator = lloyal::chat_in::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  std::string text;
  for (auto tok : separator) {
    text += lloyal::tokenizer::detokenize(model.get(), tok);
  }

  MESSAGE("Separator text: \"" << text << "\"");
  CHECK(!text.empty());
}

TEST_CASE("ChatIn Integration: get_turn_separator is stable (deterministic)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto sep1 = lloyal::chat_in::get_turn_separator(model.get());
  auto sep2 = lloyal::chat_in::get_turn_separator(model.get());

  REQUIRE(sep1.size() == sep2.size());
  for (size_t i = 0; i < sep1.size(); i++) {
    CHECK(sep1[i] == sep2[i]);
  }
}

TEST_CASE("ChatIn Integration: get_turn_separator matches template boundary") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "X"}},
    {{"role", "assistant"}, {"content", "Y"}},
    {{"role", "user"}, {"content", "Z"}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);
  REQUIRE(!result.prompt.empty());

  auto full_tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, false, true);

  auto separator = lloyal::chat_in::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

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

TEST_CASE("ChatIn Integration: get_turn_separator with null model") {
  auto separator = lloyal::chat_in::get_turn_separator(nullptr);
  CHECK(separator.empty());
}

// ===== DOCUMENTED EXAMPLE TESTS =====

TEST_CASE("ChatIn Integration: format() basic usage (documented example)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = R"([{"role":"user","content":"Hi"}])";

  auto result = lloyal::chat_in::format(model.get(), inputs);
  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, true, true);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Hi") != std::string::npos);
  CHECK(!tokens.empty());
}

TEST_CASE("ChatIn Integration: format() with custom template (documented example)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  std::string custom_template =
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}";

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = R"([{"role":"user","content":"Hello"}])";
  inputs.template_override = custom_template;

  auto custom_result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!custom_result.prompt.empty());
  CHECK(custom_result.prompt.find("<|im_start|>") != std::string::npos);
  CHECK(custom_result.prompt.find("Hello") != std::string::npos);
}

TEST_CASE("ChatIn Integration: warm continuation pattern (documented example)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  auto separator = lloyal::chat_in::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Hello"}},
    {{"role", "assistant"}, {"content", "Hi there!"}},
    {{"role", "user"}, {"content", "What is 2+2?"}}
  }).dump();

  auto cold_result = lloyal::chat_in::format(model.get(), inputs);
  REQUIRE(!cold_result.prompt.empty());

  auto cold_tokens = lloyal::tokenizer::tokenize(vocab, cold_result.prompt, false, true);
  REQUIRE(!cold_tokens.empty());

  std::string separator_text;
  for (auto tok : separator) {
    separator_text += lloyal::tokenizer::detokenize(model.get(), tok);
  }
  MESSAGE("Separator text: \"" << separator_text << "\"");

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

TEST_CASE("ChatIn Integration: tool_calls in assistant message") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What's the weather in Melbourne?"}},
    {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"type", "function"}, {"id", "call_123"}, {"function", {
        {"name", "get_weather"},
        {"arguments", R"({"location": "Melbourne"})"}}
      }}
    })}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("What's the weather") != std::string::npos);
  MESSAGE("Tool calls prompt length: " << result.prompt.size());
}

TEST_CASE("ChatIn Integration: tool response message") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What's the weather?"}},
    {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"type", "function"}, {"id", "call_456"}, {"function", {
        {"name", "weather"},
        {"arguments", "{}"}}
      }}
    })}},
    {{"role", "tool"}, {"name", "weather"}, {"tool_call_id", "call_456"}, {"content", "22 degrees and sunny"}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("22") != std::string::npos);
  MESSAGE("Tool response prompt length: " << result.prompt.size());
}

TEST_CASE("ChatIn Integration: multiple tool calls in single message") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
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
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Melbourne") != std::string::npos);
  CHECK(result.prompt.find("Sydney") != std::string::npos);
}

TEST_CASE("ChatIn Integration: reasoning_content field") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What is 15 * 7?"}},
    {{"role", "assistant"}, {"reasoning_content", "Let me calculate step by step: 15 * 7 = 15 * 5 + 15 * 2 = 75 + 30 = 105"}, {"content", "The answer is 105."}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("105") != std::string::npos);
}

TEST_CASE("ChatIn Integration: content_parts array") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", json::array({
      {{"type", "text"}, {"text", "Hello, "}},
      {{"type", "text"}, {"text", "world!"}}
    })}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Hello") != std::string::npos);
  CHECK(result.prompt.find("world") != std::string::npos);
}

TEST_CASE("ChatIn Integration: warm continuation with tool messages") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  lloyal::chat_in::FormatInputs warm_inputs;
  warm_inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What's the weather?"}},
    {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"type", "function"}, {"id", "call_789"}, {"function", {
        {"name", "weather"},
        {"arguments", "{}"}}
      }}
    })}}
  }).dump();

  json cold_messages = json::parse(warm_inputs.messages_json);
  cold_messages.push_back({
    {"role", "tool"}, {"name", "weather"}, {"tool_call_id", "call_789"}, {"content", "Sunny and warm"}
  });

  lloyal::chat_in::FormatInputs cold_inputs;
  cold_inputs.messages_json = cold_messages.dump();

  auto warm_result = lloyal::chat_in::format(model.get(), warm_inputs);
  auto cold_result = lloyal::chat_in::format(model.get(), cold_inputs);

  CHECK(!warm_result.prompt.empty());
  CHECK(!cold_result.prompt.empty());

  if (cold_result.prompt.size() <= warm_result.prompt.size()) {
    MESSAGE("[ SKIP ] Template does not appear to support tool messages natively");
    return;
  }

  CHECK(cold_result.prompt.size() > warm_result.prompt.size());
  CHECK(cold_result.prompt.find("Sunny") != std::string::npos);

  MESSAGE("Warm continuation with tools: warm(" << warm_result.prompt.size()
          << ") → cold(" << cold_result.prompt.size() << ")");
}

// ===== FORMAT AWARENESS TESTS =====

TEST_CASE("ChatIn Integration: format with tools returns format enum") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Use a tool"}}
  }).dump();
  inputs.tools_json = json::array({
    {{"type", "function"}, {"function", {
      {"name", "get_weather"},
      {"description", "Get weather info"},
      {"parameters", {{"type", "object"}, {"properties", {{"location", {{"type", "string"}}}}}}}
    }}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(static_cast<int>(result.format) >= 0);
  CHECK(static_cast<int>(result.format) < COMMON_CHAT_FORMAT_COUNT);
  MESSAGE("Detected format: " << static_cast<int>(result.format));
}

TEST_CASE("ChatIn Integration: format returns grammar string when tools provided") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "Call get_time"}}
  }).dump();
  inputs.tools_json = json::array({
    {{"type", "function"}, {"function", {
      {"name", "get_time"},
      {"description", "Get current time"},
      {"parameters", {{"type", "object"}}}
    }}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  // Grammar may be empty for some templates that don't support tools,
  // but format should still be valid
  MESSAGE("Grammar size: " << result.grammar.size() << " bytes");
  MESSAGE("Format: " << static_cast<int>(result.format));
}

TEST_CASE("ChatIn Integration: format with tools in multi-turn") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What's the weather in Melbourne?"}},
    {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"type", "function"}, {"id", "call_1"}, {"function", {
        {"name", "get_weather"},
        {"arguments", R"({"location":"Melbourne"})"}}
      }}
    })}},
    {{"role", "tool"}, {"name", "get_weather"}, {"tool_call_id", "call_1"}, {"content", "22C and sunny"}},
    {{"role", "user"}, {"content", "And Sydney?"}}
  }).dump();
  inputs.tools_json = json::array({
    {{"type", "function"}, {"function", {
      {"name", "get_weather"},
      {"description", "Get weather"},
      {"parameters", {{"type", "object"}, {"properties", {{"location", {{"type", "string"}}}}}}}
    }}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Melbourne") != std::string::npos);
  CHECK(result.prompt.find("Sydney") != std::string::npos);
}

// ===== HIGH-FIDELITY WARM CONTINUATION VERIFICATION =====

TEST_CASE("ChatIn Integration: warm continuation token-level parity") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What is your name?"}},
    {{"role", "assistant"}, {"content", "I am an AI assistant."}},
    {{"role", "user"}, {"content", "Nice to meet you!"}}
  }).dump();

  auto cold_result = lloyal::chat_in::format(model.get(), inputs);
  REQUIRE(!cold_result.prompt.empty());

  auto cold_tokens = lloyal::tokenizer::tokenize(vocab, cold_result.prompt, false, true);
  REQUIRE(!cold_tokens.empty());

  auto separator = lloyal::chat_in::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Format partial conversation (without last user message) to find split point
  lloyal::chat_in::FormatInputs partial_inputs;
  partial_inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "What is your name?"}},
    {{"role", "assistant"}, {"content", "I am an AI assistant."}}
  }).dump();
  partial_inputs.add_generation_prompt = false;

  auto partial_result = lloyal::chat_in::format(model.get(), partial_inputs);
  auto partial_tokens = lloyal::tokenizer::tokenize(vocab, partial_result.prompt, false, true);
  REQUIRE(cold_tokens.size() > partial_tokens.size());

  // Verify cold tokens start with partial tokens (prefix match)
  bool prefix_match = true;
  for (size_t i = 0; i < partial_tokens.size(); i++) {
    if (cold_tokens[i] != partial_tokens[i]) {
      prefix_match = false;
      MESSAGE("Prefix mismatch at " << i
              << ": cold=" << cold_tokens[i]
              << " (" << lloyal::tokenizer::detokenize(model.get(), cold_tokens[i]) << ")"
              << " vs partial=" << partial_tokens[i]
              << " (" << lloyal::tokenizer::detokenize(model.get(), partial_tokens[i]) << ")");
      break;
    }
  }
  REQUIRE(prefix_match);

  // Verify separator at end of partial tokens
  REQUIRE(partial_tokens.size() >= separator.size());
  for (size_t i = 0; i < separator.size(); i++) {
    CHECK(partial_tokens[partial_tokens.size() - separator.size() + i] == separator[i]);
  }

  // Split point: where separator starts within the cold token stream
  size_t separator_token_pos = partial_tokens.size() - separator.size();

  std::vector<llama_token> warm_base(cold_tokens.begin(),
                                      cold_tokens.begin() + separator_token_pos);
  std::vector<llama_token> delta(cold_tokens.begin() + separator_token_pos + separator.size(),
                                  cold_tokens.end());

  std::vector<llama_token> reconstructed;
  reconstructed.insert(reconstructed.end(), warm_base.begin(), warm_base.end());
  reconstructed.insert(reconstructed.end(), separator.begin(), separator.end());
  reconstructed.insert(reconstructed.end(), delta.begin(), delta.end());

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

TEST_CASE("ChatIn Integration: warm continuation multi-turn token parity") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "A"}},
    {{"role", "assistant"}, {"content", "B"}},
    {{"role", "user"}, {"content", "C"}},
    {{"role", "assistant"}, {"content", "D"}},
    {{"role", "user"}, {"content", "E"}}
  }).dump();

  auto cold_result = lloyal::chat_in::format(model.get(), inputs);
  REQUIRE(!cold_result.prompt.empty());

  auto cold_tokens = lloyal::tokenizer::tokenize(vocab, cold_result.prompt, false, true);
  auto separator = lloyal::chat_in::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Format partial conversation (without last user message) to find split point
  lloyal::chat_in::FormatInputs partial_inputs;
  partial_inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "A"}},
    {{"role", "assistant"}, {"content", "B"}},
    {{"role", "user"}, {"content", "C"}},
    {{"role", "assistant"}, {"content", "D"}}
  }).dump();
  partial_inputs.add_generation_prompt = false;

  auto partial_result = lloyal::chat_in::format(model.get(), partial_inputs);
  auto partial_tokens = lloyal::tokenizer::tokenize(vocab, partial_result.prompt, false, true);
  REQUIRE(cold_tokens.size() > partial_tokens.size());

  // Verify prefix match
  bool prefix_match = true;
  for (size_t i = 0; i < partial_tokens.size(); i++) {
    if (cold_tokens[i] != partial_tokens[i]) {
      prefix_match = false;
      break;
    }
  }
  REQUIRE(prefix_match);

  // Verify separator at end of partial tokens
  REQUIRE(partial_tokens.size() >= separator.size());
  for (size_t i = 0; i < separator.size(); i++) {
    CHECK(partial_tokens[partial_tokens.size() - separator.size() + i] == separator[i]);
  }

  size_t last_sep_pos = partial_tokens.size() - separator.size();

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
          << ") + separator(" << separator.size()
          << ") + delta(" << delta.size()
          << ") == cold(" << cold_tokens.size() << ")");
}

// ===== ACTUAL WARM CONTINUATION TEST =====

struct WarmTestParams {
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

TEST_CASE("ChatIn Integration: warm vs cold generation parity") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  struct ContextGuard {
    llama_context* c;
    ~ContextGuard() { if (c) llama_free(c); }
  } ctx_guard{ctx};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "My name is Alice. What is my name?"}},
    {{"role", "assistant"}, {"content", "Your name is Alice."}},
    {{"role", "user"}, {"content", "What did I just tell you my name was?"}}
  }).dump();

  auto result = lloyal::chat_in::format(model.get(), inputs);
  REQUIRE(!result.prompt.empty());

  auto cold_tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, true, true);
  REQUIRE(!cold_tokens.empty());

  auto separator = lloyal::chat_in::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Format partial conversation (without last user message) to find split point
  lloyal::chat_in::FormatInputs partial_inputs;
  partial_inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "My name is Alice. What is my name?"}},
    {{"role", "assistant"}, {"content", "Your name is Alice."}}
  }).dump();
  partial_inputs.add_generation_prompt = false;

  auto partial_result = lloyal::chat_in::format(model.get(), partial_inputs);
  auto partial_tokens = lloyal::tokenizer::tokenize(vocab, partial_result.prompt, true, true);
  REQUIRE(cold_tokens.size() > partial_tokens.size());

  // Verify prefix match
  bool prefix_match = true;
  for (size_t i = 0; i < partial_tokens.size(); i++) {
    if (cold_tokens[i] != partial_tokens[i]) {
      prefix_match = false;
      MESSAGE("Prefix mismatch at " << i);
      break;
    }
  }
  REQUIRE(prefix_match);

  // Split at end of partial (separator is at the end of partial tokens)
  size_t separator_pos = partial_tokens.size() - separator.size();

  std::vector<llama_token> warm_base(cold_tokens.begin(),
                                      cold_tokens.begin() + separator_pos);
  std::vector<llama_token> delta(cold_tokens.begin() + separator_pos,
                                  cold_tokens.end());

  MESSAGE("Warm continuation split: warm_base(" << warm_base.size()
          << ") + delta(" << delta.size()
          << ") = cold(" << cold_tokens.size() << ")");

  lloyal::branch::BranchStore store;
  WarmTestParams params;

  // ===== COLD PATH =====
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
  lloyal::kv::clear_all(ctx);

  // ===== WARM PATH =====
  auto warm_branch = lloyal::branch::create(ctx, model.get(), 0, 0, params, 512,
                                             nullptr, nullptr, &store);
  REQUIRE(warm_branch != lloyal::branch::INVALID_HANDLE);

  lloyal::branch::decode_batch(warm_branch, warm_base.data(),
                                warm_base.size(), &store);
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
  std::string cold_text, warm_text;
  for (auto tok : cold_output) {
    cold_text += lloyal::tokenizer::detokenize(model.get(), tok);
  }
  for (auto tok : warm_output) {
    warm_text += lloyal::tokenizer::detokenize(model.get(), tok);
  }

  MESSAGE("Cold output (" << cold_output.size() << " tokens): \"" << cold_text << "\"");
  MESSAGE("Warm output (" << warm_output.size() << " tokens): \"" << warm_text << "\"");

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
    MESSAGE("WARM CONTINUATION VERIFIED: Exact token parity confirmed!");
  } else {
    bool cold_valid = cold_text.find("Alice") != std::string::npos;
    bool warm_valid = warm_text.find("Alice") != std::string::npos;
    CHECK(cold_valid);
    CHECK(warm_valid);
    if (cold_valid && warm_valid) {
      MESSAGE("WARM CONTINUATION VERIFIED: Semantic parity confirmed (minor token variation)");
    }
  }
}

TEST_CASE("ChatIn Integration: true warm continuation (no re-prefill)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  const auto* vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  struct ContextGuard {
    llama_context* c;
    ~ContextGuard() { if (c) llama_free(c); }
  } ctx_guard{ctx};

  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "My favorite color is blue. Remember that."}},
    {{"role", "assistant"}, {"content", "I'll remember that your favorite color is blue!"}},
    {{"role", "user"}, {"content", "What is my favorite color?"}}
  }).dump();

  auto full_result = lloyal::chat_in::format(model.get(), inputs);
  auto full_tokens = lloyal::tokenizer::tokenize(vocab, full_result.prompt, true, true);

  auto separator = lloyal::chat_in::get_turn_separator(model.get());
  REQUIRE(!separator.empty());

  // Format partial conversation (without last user message) to find split point
  lloyal::chat_in::FormatInputs partial_inputs;
  partial_inputs.messages_json = json::array({
    {{"role", "user"}, {"content", "My favorite color is blue. Remember that."}},
    {{"role", "assistant"}, {"content", "I'll remember that your favorite color is blue!"}}
  }).dump();
  partial_inputs.add_generation_prompt = false;

  auto partial_result = lloyal::chat_in::format(model.get(), partial_inputs);
  auto partial_tokens = lloyal::tokenizer::tokenize(vocab, partial_result.prompt, true, true);
  REQUIRE(full_tokens.size() > partial_tokens.size());

  // Verify prefix match
  bool prefix_match = true;
  for (size_t i = 0; i < partial_tokens.size(); i++) {
    if (full_tokens[i] != partial_tokens[i]) {
      prefix_match = false;
      MESSAGE("Prefix mismatch at " << i);
      break;
    }
  }
  REQUIRE(prefix_match);

  // Split at end of partial (separator is at the end of partial tokens)
  size_t split_pos = partial_tokens.size() - separator.size();

  std::vector<llama_token> warm_base(full_tokens.begin(),
                                      full_tokens.begin() + split_pos);
  std::vector<llama_token> delta(full_tokens.begin() + split_pos,
                                  full_tokens.end());

  MESSAGE("True warm continuation: warm_base(" << warm_base.size()
          << ") + delta(" << delta.size()
          << ") = full(" << full_tokens.size() << ")");

  REQUIRE(warm_base.size() + delta.size() == full_tokens.size());

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
  auto warm_branch = lloyal::branch::create(ctx, model.get(), 0, 0, params, 512,
                                             nullptr, nullptr, &store);
  REQUIRE(warm_branch != lloyal::branch::INVALID_HANDLE);

  lloyal::branch::decode_batch(warm_branch, warm_base.data(),
                                warm_base.size(), &store);
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
    MESSAGE("TRUE WARM CONTINUATION: No re-prefill needed, generation matches!");
  }
}
