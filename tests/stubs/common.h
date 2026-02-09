// Stub for llama.cpp common/common.h
// Provides batch utilities used by decoder.hpp and embedding.hpp

#pragma once

#include "llama/llama.h"
#include <vector>

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
