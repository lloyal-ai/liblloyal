// Stub for llama.cpp common/common.h
// Provides the grammar trigger and reasoning format types used by liblloyal headers

#pragma once

#include "llama/llama.h"
#include <string>
#include <vector>

// Grammar trigger types (from llama.cpp common/common.h)
enum common_grammar_trigger_type {
  COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
  COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
  COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
  COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

struct common_grammar_trigger {
  common_grammar_trigger_type type;
  std::string value;
  llama_token token = LLAMA_TOKEN_NULL;
};

// Reasoning format enum (from llama.cpp common/common.h)
enum common_reasoning_format {
  COMMON_REASONING_FORMAT_NONE,
  COMMON_REASONING_FORMAT_AUTO,
  COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
  COMMON_REASONING_FORMAT_DEEPSEEK,
};
