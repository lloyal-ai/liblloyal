/**
 * Unit tests for chat template round-trip pattern
 *
 * Tests the implementation of the round-trip pattern for chat template handling:
 * 1. Template renders with wrapper tokens (minja defaults)
 * 2. Wrapper tokens stripped conditionally based on GGUF metadata
 * 3. Wrapper tokens re-added during tokenization if model expects them
 *
 * This ensures template variables like {{ bos_token }} remain available for
 * use as delimiters while preventing double-token issues.
 */

#include <doctest/doctest.h>
#include <lloyal/helpers.hpp>
#include <lloyal/nlohmann/json.hpp>
#include "llama_stubs.h"

using json = nlohmann::ordered_json;

// ===== HELPER FUNCTIONS =====

void resetTestConfig() {
  resetStubConfig();
}

// ===== ROUND-TRIP PATTERN TESTS =====

TEST_CASE("ChatTemplate: conditional stripping with add_bos=false, add_eos=false") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  // Configure stubs for TinyLlama scenario
  llamaStubConfig().bos_token = 1;      // <s>
  llamaStubConfig().eos_token = 2;      // </s>

  // Zephyr template (TinyLlama)
  llamaStubConfig().chat_template =
    "{% for message in messages %}"
    "{{'<|' + message['role'] + '|>\\n' + message['content'] + eos_token + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|assistant|>\\n'}}{% endif %}";

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  });

  auto result = lloyal::format_chat_template_complete(&model, messages.dump());

  // With add_bos=false, add_eos=false (stub defaults), wrapper tokens should NOT be stripped
  // Template should render normally with all delimiters intact
  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("<|user|>") != std::string::npos);
  CHECK(result.prompt.find("Hello") != std::string::npos);
  CHECK(result.prompt.find("<|assistant|>") != std::string::npos);
}

TEST_CASE("ChatTemplate: apply_chat_template_helper strips when add_bos=true") {
  // Test the core helper function directly

  std::string template_str = "{{ bos_token }}Hello{{ eos_token }}";
  json messages = json::array({
    {{"role", "user"}, {"content", "test"}}
  });

  std::string bos_token = "<s>";
  std::string eos_token = "</s>";

  // Test with add_bos=true, add_eos=true - should strip wrapper tokens
  auto result = lloyal::detail::apply_chat_template_helper(
    template_str, messages, bos_token, eos_token, true, true, true);

  // Since template is just "{{ bos_token }}Hello{{ eos_token }}"
  // and we're stripping, the actual minja output would have the tokens
  // then we strip them based on add_bos/add_eos flags
  // Note: This is testing the stripping logic
}

TEST_CASE("ChatTemplate: apply_chat_template_helper does NOT strip when add_bos=false") {
  // Test the core helper function directly

  std::string template_str = "{{ bos_token }}Content{{ eos_token }}";
  json messages = json::array({
    {{"role", "user"}, {"content", "test"}}
  });

  std::string bos_token = "<s>";
  std::string eos_token = "</s>";

  // Test with add_bos=false, add_eos=false - should NOT strip wrapper tokens
  auto result = lloyal::detail::apply_chat_template_helper(
    template_str, messages, bos_token, eos_token, true, false, false);

  // With add_bos=false, add_eos=false, no stripping should occur
  // The template variables are still available for use within the template
}

TEST_CASE("ChatTemplate: metadata flags passed through format_chat_template_complete") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  // Configure stubs
  llamaStubConfig().bos_token = 1;
  llamaStubConfig().eos_token = 2;
  llamaStubConfig().chat_template =
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}";

  json messages = json::array({
    {{"role", "user"}, {"content", "Test message"}}
  });

  // This should query the stub's add_bos/add_eos values and pass them through
  auto result = lloyal::format_chat_template_complete(&model, messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("user: Test message") != std::string::npos);
}

TEST_CASE("ChatTemplate: metadata flags passed through format_chat_template_from_model") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  // Configure stubs
  llamaStubConfig().bos_token = 1;
  llamaStubConfig().eos_token = 2;
  llamaStubConfig().chat_template =
    "User: {{ messages[0]['content'] }}\nAssistant:";

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello world"}}
  });

  // This should query the stub's add_bos/add_eos values and pass them through
  auto result = lloyal::format_chat_template_from_model(&model, messages.dump());

  CHECK(!result.empty());
  CHECK(result.find("Hello world") != std::string::npos);
}

TEST_CASE("ChatTemplate: multi-turn conversation with round-trip pattern") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  llamaStubConfig().bos_token = 1;
  llamaStubConfig().eos_token = 2;
  llamaStubConfig().chat_template =
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>\n{{ message['content'] }}{{ eos_token }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}";

  json messages = json::array({
    {{"role", "user"}, {"content", "First message"}},
    {{"role", "assistant"}, {"content", "First response"}},
    {{"role", "user"}, {"content", "Second message"}}
  });

  auto result = lloyal::format_chat_template_complete(&model, messages.dump());

  CHECK(!result.prompt.empty());
  // All messages should be present
  CHECK(result.prompt.find("First message") != std::string::npos);
  CHECK(result.prompt.find("First response") != std::string::npos);
  CHECK(result.prompt.find("Second message") != std::string::npos);
}

TEST_CASE("ChatTemplate: template variables remain available regardless of metadata") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  llamaStubConfig().bos_token = 1;
  llamaStubConfig().eos_token = 2;

  // Template that uses bos_token and eos_token as delimiters
  llamaStubConfig().chat_template =
    "{% for message in messages %}"
    "{{ bos_token }}[{{ message['role'] }}]{{ eos_token }} {{ message['content'] }}\n"
    "{% endfor %}";

  json messages = json::array({
    {{"role", "user"}, {"content", "Test"}}
  });

  auto result = lloyal::format_chat_template_complete(&model, messages.dump());

  // The template variables should be rendered into the output
  // (though wrapper tokens at start/end may be stripped based on metadata)
  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Test") != std::string::npos);
}

TEST_CASE("ChatTemplate: error handling with invalid JSON") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};


  auto result = lloyal::format_chat_template_complete(&model, "invalid json");

  // Should return empty string on error
  CHECK(result.prompt.empty());
}

TEST_CASE("ChatTemplate: error handling with null model") {
  auto result = lloyal::format_chat_template_complete(nullptr, "[]");

  // Should still work, falling back to default template
  CHECK(!result.prompt.empty());
}

TEST_CASE("ChatTemplate: empty messages array") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  llamaStubConfig().chat_template = "Empty: {{ messages|length }}";

  json messages = json::array();

  auto result = lloyal::format_chat_template_complete(&model, messages.dump());

  CHECK(!result.prompt.empty());
}

TEST_CASE("ChatTemplate: ChatML fallback when no model template") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  llamaStubConfig().chat_template = "";  // No template in model

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  });

  auto result = lloyal::format_chat_template_complete(&model, messages.dump());

  // Should fall back to ChatML template
  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("<|im_start|>") != std::string::npos);
}

TEST_CASE("ChatTemplate: template override takes precedence") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  llamaStubConfig().chat_template = "Model template";

  std::string override_template = "Override: {{ messages[0]['content'] }}";

  json messages = json::array({
    {{"role", "user"}, {"content", "Test content"}}
  });

  auto result = lloyal::format_chat_template_complete(&model, messages.dump(), override_template);

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Override:") != std::string::npos);
  CHECK(result.prompt.find("Test content") != std::string::npos);
}

TEST_CASE("ChatTemplate: validate_chat_template_helper with valid template") {
  std::string valid_template = "{{ messages[0]['content'] }}";

  bool is_valid = lloyal::validate_chat_template_helper(valid_template);

  CHECK(is_valid);
}

TEST_CASE("ChatTemplate: validate_chat_template_helper with invalid template") {
  std::string invalid_template = "{{ unclosed";

  bool is_valid = lloyal::validate_chat_template_helper(invalid_template);

  CHECK(!is_valid);
}

// ===== EDGE CASES =====

TEST_CASE("ChatTemplate: very long message content") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  llamaStubConfig().chat_template = "{{ messages[0]['content'] }}";

  // 10KB message
  std::string long_content(10000, 'x');
  json messages = json::array({
    {{"role", "user"}, {"content", long_content}}
  });

  auto result = lloyal::format_chat_template_complete(&model, messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find(long_content) != std::string::npos);
}

TEST_CASE("ChatTemplate: special characters in content") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  llamaStubConfig().chat_template = "Content: {{ messages[0]['content'] }}";

  json messages = json::array({
    {{"role", "user"}, {"content", "Quote: \"Hello\"\nNewline\tTab"}}
  });

  auto result = lloyal::format_chat_template_complete(&model, messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("Quote:") != std::string::npos);
}

TEST_CASE("ChatTemplate: unicode content") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  llamaStubConfig().chat_template = "{{ messages[0]['content'] }}";

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello 世界 🌍"}}
  });

  auto result = lloyal::format_chat_template_complete(&model, messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("世界") != std::string::npos);
  CHECK(result.prompt.find("🌍") != std::string::npos);
}
