/**
 * Unit tests for chat template processing
 *
 * Tests the chat template API using llama.cpp's common library:
 * - Template formatting with fallback
 * - Template validation
 * - Turn separator extraction
 *
 * Note: This test uses stubs since we can't load real models in unit tests.
 * For full integration testing, see chat_template_integration_test.cpp
 */

#include <doctest/doctest.h>
#include <lloyal/chat_template.hpp>
#include <nlohmann/json.hpp>
#include "llama_stubs.h"

using json = nlohmann::ordered_json;

// ===== HELPER FUNCTIONS =====

void resetTestConfig() {
  resetStubConfig();
}

// ===== BASIC FORMATTING TESTS =====

TEST_CASE("ChatTemplate: format returns FormatResult with prompt") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello"}}
  });

  auto result = lloyal::chat_template::format(&model, messages.dump());

  // Should return a result with prompt
  // Note: Without a real model, this may fall back to simple format
  CHECK(result.prompt.find("Hello") != std::string::npos);
}

TEST_CASE("ChatTemplate: format with empty messages") {
  resetTestConfig();

  llama_model model{};

  json messages = json::array();

  auto result = lloyal::chat_template::format(&model, messages.dump());

  // Empty messages should produce empty or minimal result
  // No stop tokens should be extracted from empty input
  CHECK(result.additional_stops.empty());
  // Prompt may be empty or contain just generation prompt prefix (model-dependent)
  // The key contract: it doesn't crash and returns a valid FormatResult
}

TEST_CASE("ChatTemplate: format with null model") {
  json messages = json::array({
    {{"role", "user"}, {"content", "Test"}}
  });

  auto result = lloyal::chat_template::format(nullptr, messages.dump());

  // Should fall back gracefully
  CHECK(result.prompt.find("Test") != std::string::npos);
}

TEST_CASE("ChatTemplate: format with invalid JSON") {
  resetTestConfig();

  llama_model model{};

  auto result = lloyal::chat_template::format(&model, "invalid json");

  // Should return empty result on error
  CHECK(result.prompt.empty());
}

// ===== VALIDATION TESTS =====

TEST_CASE("ChatTemplate: validate with valid template") {
  std::string valid_template = "{{ messages[0]['content'] }}";

  bool is_valid = lloyal::chat_template::validate(valid_template);

  CHECK(is_valid);
}

TEST_CASE("ChatTemplate: validate with invalid template") {
  std::string invalid_template = "{{ unclosed";

  bool is_valid = lloyal::chat_template::validate(invalid_template);

  CHECK(!is_valid);
}

// ===== MULTI-MESSAGE TESTS =====

TEST_CASE("ChatTemplate: format multi-turn conversation") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  json messages = json::array({
    {{"role", "user"}, {"content", "First message"}},
    {{"role", "assistant"}, {"content", "First response"}},
    {{"role", "user"}, {"content", "Second message"}}
  });

  auto result = lloyal::chat_template::format(&model, messages.dump());

  // All messages should be present in some form
  CHECK(result.prompt.find("First message") != std::string::npos);
  CHECK(result.prompt.find("First response") != std::string::npos);
  CHECK(result.prompt.find("Second message") != std::string::npos);
}

// ===== EDGE CASES =====

TEST_CASE("ChatTemplate: format with very long message") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  // 10KB message
  std::string long_content(10000, 'x');
  json messages = json::array({
    {{"role", "user"}, {"content", long_content}}
  });

  auto result = lloyal::chat_template::format(&model, messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find(long_content) != std::string::npos);
}

TEST_CASE("ChatTemplate: format with special characters") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  json messages = json::array({
    {{"role", "user"}, {"content", "Quote: \"Hello\"\nNewline\tTab"}}
  });

  auto result = lloyal::chat_template::format(&model, messages.dump());

  CHECK(!result.prompt.empty());
}

TEST_CASE("ChatTemplate: format with unicode content") {
  resetTestConfig();

  llama_model model{};
  llama_vocab vocab{};

  json messages = json::array({
    {{"role", "user"}, {"content", "Hello 世界 🌍"}}
  });

  auto result = lloyal::chat_template::format(&model, messages.dump());

  CHECK(!result.prompt.empty());
  CHECK(result.prompt.find("世界") != std::string::npos);
}

// ===== FALLBACK_TO_EOG TESTS =====

TEST_CASE("ChatTemplate: fallback_to_eog with null model") {
  auto result = lloyal::chat_template::fallback_to_eog(nullptr);

  CHECK(result.empty());
}
