// Stub for llama.cpp common/common.h
// Provides batch utilities and grammar trigger types used by liblloyal headers

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

// Batch utilities - stub implementations
inline void common_batch_clear(llama_batch& batch) {
  batch.n_tokens = 0;
}

inline void common_batch_add(
    llama_batch& batch,
    llama_token token,
    llama_pos pos,
    const std::vector<llama_seq_id>& seq_ids,
    bool logits
) {
  if (batch.token) batch.token[batch.n_tokens] = token;
  if (batch.pos) batch.pos[batch.n_tokens] = pos;
  if (batch.n_seq_id) batch.n_seq_id[batch.n_tokens] = static_cast<int32_t>(seq_ids.size());
  if (batch.seq_id && !seq_ids.empty()) {
    batch.seq_id[batch.n_tokens][0] = seq_ids[0];
  }
  if (batch.logits) batch.logits[batch.n_tokens] = logits ? 1 : 0;
  batch.n_tokens++;
}
