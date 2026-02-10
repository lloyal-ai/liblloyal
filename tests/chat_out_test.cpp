/**
 * Unit tests for chat output parsing
 *
 * Tests the chat_out API using stubs:
 * - Plain content passthrough
 * - Format-specific parsing behavior
 * - Reasoning format forwarding
 * - Partial output handling
 * - Error handling
 *
 * Note: Stub's common_chat_parse() returns content passthrough.
 * Real parsing behavior is tested in chat_out_integration_test.cpp.
 */

#include <doctest/doctest.h>
#include <lloyal/chat_out.hpp>
#include "llama_stubs.h"

// ===== BASIC PARSING TESTS =====

TEST_CASE("ChatOut: parse plain content returns content") {
  auto result = lloyal::chat_out::parse(
    "Hello, world!",
    COMMON_CHAT_FORMAT_CONTENT_ONLY
  );

  CHECK(result.content == "Hello, world!");
  CHECK(result.reasoning_content.empty());
  CHECK(result.tool_calls.empty());
}

TEST_CASE("ChatOut: parse with CONTENT_ONLY is passthrough") {
  auto result = lloyal::chat_out::parse(
    "Some output text",
    COMMON_CHAT_FORMAT_CONTENT_ONLY
  );

  CHECK(result.content == "Some output text");
  CHECK(result.tool_calls.empty());
}

TEST_CASE("ChatOut: parse with GENERIC format") {
  auto result = lloyal::chat_out::parse(
    "Response with tools",
    COMMON_CHAT_FORMAT_GENERIC
  );

  // Stub returns passthrough regardless of format
  CHECK(result.content == "Response with tools");
}

TEST_CASE("ChatOut: parse forwards reasoning_format") {
  auto result = lloyal::chat_out::parse(
    "thinking output",
    COMMON_CHAT_FORMAT_DEEPSEEK_R1,
    COMMON_REASONING_FORMAT_DEEPSEEK
  );

  // Stub doesn't extract reasoning, but verifies no crash
  CHECK(result.content == "thinking output");
}

TEST_CASE("ChatOut: parse with is_partial true") {
  auto result = lloyal::chat_out::parse(
    "partial out",
    COMMON_CHAT_FORMAT_CONTENT_ONLY,
    COMMON_REASONING_FORMAT_NONE,
    true  // is_partial
  );

  CHECK(result.content == "partial out");
}

TEST_CASE("ChatOut: parse empty string") {
  auto result = lloyal::chat_out::parse(
    "",
    COMMON_CHAT_FORMAT_CONTENT_ONLY
  );

  CHECK(result.content.empty());
  CHECK(result.tool_calls.empty());
}

TEST_CASE("ChatOut: parse with thinking_forced_open") {
  auto result = lloyal::chat_out::parse(
    "still thinking",
    COMMON_CHAT_FORMAT_DEEPSEEK_R1,
    COMMON_REASONING_FORMAT_DEEPSEEK,
    false,  // is_partial
    true    // thinking_forced_open
  );

  // Stub passthrough — just verify no crash
  CHECK(result.content == "still thinking");
}

// ===== MODEL-BASED OVERLOAD TESTS =====

TEST_CASE("ChatOut: parse with model auto-detects format") {
  llama_model model{};

  auto result = lloyal::chat_out::parse(
    &model,
    "Model output text"
  );

  // Stub returns CONTENT_ONLY format → passthrough
  CHECK(result.content == "Model output text");
  CHECK(result.tool_calls.empty());
}

TEST_CASE("ChatOut: parse with null model returns raw output") {
  auto result = lloyal::chat_out::parse(
    static_cast<const llama_model*>(nullptr),
    "Raw output"
  );

  CHECK(result.content == "Raw output");
}

TEST_CASE("ChatOut: parse with model and is_partial") {
  llama_model model{};

  auto result = lloyal::chat_out::parse(
    &model,
    "Partial model output",
    true  // is_partial
  );

  CHECK(result.content == "Partial model output");
}

// ===== RESULT STRUCTURE TESTS =====

TEST_CASE("ChatOut: ParseResult has expected fields") {
  lloyal::chat_out::ParseResult result;

  CHECK(result.content.empty());
  CHECK(result.reasoning_content.empty());
  CHECK(result.tool_calls.empty());
}

TEST_CASE("ChatOut: ToolCall has expected fields") {
  lloyal::chat_out::ToolCall tc;
  tc.name = "get_weather";
  tc.arguments = R"({"location":"Melbourne"})";
  tc.id = "call_123";

  CHECK(tc.name == "get_weather");
  CHECK(tc.arguments == R"({"location":"Melbourne"})");
  CHECK(tc.id == "call_123");
}
