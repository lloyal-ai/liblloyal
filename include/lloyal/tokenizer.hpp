#pragma once

#include "common.hpp"
#include <cstdint>
#include <llama/llama.h>
#include <string>
#include <vector>

/**
 * Tokenizer Anti-Corruption Layer (Header-Only)
 *
 * Purpose: Single point of contact with llama.cpp tokenization APIs to isolate
 * version churn, special token handling complexity, and buffer sizing edge
 * cases.
 *
 * Uses two-pass algorithms for safe buffer sizing.
 */

namespace lloyal::tokenizer {

// ===== TOKENIZATION (TEXT → TOKENS) =====

/**
 * Tokenize text to token array
 *
 * @param vocab Vocabulary from llama_model_get_vocab()
 * @param text Text to tokenize
 * @param add_special Add special tokens (BOS/EOS) if model configured
 * @param parse_special Parse special token strings like "<|im_start|>"
 * @return Vector of token IDs
 */
inline std::vector<llama_token> tokenize(const llama_vocab *vocab,
                                         const std::string &text,
                                         bool add_special, bool parse_special) {
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[tokenizer::tokenize] ERROR: vocab is null");
    return {};
  }

  // Two-pass algorithm for safety:
  // Pass 1: Determine required buffer size (negative return = size needed)
  const int n_tokens =
      -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.length()),
                      nullptr, // null buffer to get size
                      0, add_special, parse_special);

  if (n_tokens <= 0) {
    LLOYAL_LOG_DEBUG("[tokenizer::tokenize] WARNING: Empty tokenization result "
                     "for text: '%.50s...'",
                     text.c_str());
    return {};
  }

  // Pass 2: Actual tokenization
  std::vector<llama_token> tokens(n_tokens);
  const int n_tokenized =
      llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.length()),
                     tokens.data(), n_tokens, add_special, parse_special);

  if (n_tokenized != n_tokens) {
    LLOYAL_LOG_DEBUG("[tokenizer::tokenize] ERROR: Token count mismatch "
                     "(expected %d, got %d)",
                     n_tokens, n_tokenized);
    return {};
  }

  LLOYAL_LOG_DEBUG("[tokenizer::tokenize] Tokenized %zu bytes → %d tokens",
                   text.length(), n_tokens);
  return tokens;
}

// ===== DETOKENIZATION (TOKENS → TEXT) =====

/**
 * Detokenize SINGLE token to text (streaming use case)
 *
 * Fast synchronous operation for per-token conversion during generation.
 * AVOID CONFUSION: This is NOT llama_decode (KV cache update).
 *
 * @param vocab Vocabulary from llama_model_get_vocab()
 * @param token Token ID to convert
 * @param special Enable special token rendering (e.g., "<|im_start|>")
 * @return Text representation of token
 */
inline std::string detokenize(const llama_vocab *vocab, llama_token token,
                              bool special) {
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[tokenizer::detokenize] ERROR: vocab is null");
    return "";
  }

  // Two-pass algorithm (vendored from llama.cpp/common/common.cpp)
  std::string piece;
  piece.resize(
      piece.capacity()); // Use string's internal cache (15 bytes + '\0')

  const int n_chars =
      llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);

  if (n_chars < 0) {
    // Buffer too small, resize and retry
    piece.resize(-n_chars);
    int check =
        llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (check != -n_chars) {
      LLOYAL_LOG_DEBUG(
          "[tokenizer::detokenize] ERROR: Inconsistent sizing for token %d",
          token);
      return "";
    }
  } else {
    piece.resize(n_chars);
  }

  return piece;
}

/**
 * Detokenize TOKEN ARRAY to text (reconstruction use case)
 *
 * Batch operation for reconstructing complete text from token sequences.
 * AVOID CONFUSION: This is NOT llama_decode (KV cache update).
 *
 * @param vocab Vocabulary from llama_model_get_vocab()
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens in array
 * @param remove_special Remove BOS/EOS tokens from output
 * @param unparse_special Render special tokens as text (e.g., "<|im_start|>")
 * @return Complete text representation
 */
inline std::string detokenize_batch(const llama_vocab *vocab,
                                    const llama_token *tokens, int32_t n_tokens,
                                    bool remove_special, bool unparse_special) {
  if (!vocab || !tokens || n_tokens <= 0) {
    LLOYAL_LOG_DEBUG("[tokenizer::detokenize_batch] ERROR: Invalid parameters "
                     "(vocab=%p, tokens=%p, n_tokens=%d)",
                     vocab, tokens, n_tokens);
    return "";
  }

  // Two-pass algorithm for safety:
  // Pass 1: Determine required buffer size (negative return = size needed)
  int32_t required_size = llama_detokenize(vocab, tokens, n_tokens,
                                           nullptr, // null buffer to get size
                                           0, remove_special, unparse_special);

  if (required_size < 0) {
    // Negative return means we need abs(required_size) bytes
    required_size = -required_size;
  }

  if (required_size == 0) {
    LLOYAL_LOG_DEBUG("[tokenizer::detokenize_batch] WARNING: Empty "
                     "detokenization result for %d tokens",
                     n_tokens);
    return "";
  }

  // Pass 2: Actual detokenization
  std::vector<char> buffer(required_size + 1); // +1 for null terminator
  int32_t written =
      llama_detokenize(vocab, tokens, n_tokens, buffer.data(), required_size,
                       remove_special, unparse_special);

  if (written < 0) {
    LLOYAL_LOG_DEBUG("[tokenizer::detokenize_batch] ERROR: Detokenization "
                     "failed (needed %d bytes, got %d)",
                     required_size, written);
    return "";
  }

  std::string result(buffer.data(), written);
  LLOYAL_LOG_DEBUG(
      "[tokenizer::detokenize_batch] Detokenized %d tokens → %zu bytes",
      n_tokens, result.size());
  return result;
}

// ===== VOCABULARY QUERIES =====

/**
 * Get vocabulary from model
 *
 * Simple accessor that wraps llama_model_get_vocab().
 * Isolates direct llama.cpp model API dependency.
 *
 * @param model Llama model
 * @return Vocabulary pointer (never null if model is valid)
 */
inline const llama_vocab *get_vocab(const llama_model *model) {
  if (!model) {
    LLOYAL_LOG_DEBUG("[tokenizer::get_vocab] ERROR: NULL model");
    return nullptr;
  }

  const llama_vocab *vocab = llama_model_get_vocab(model);
  if (!vocab) {
    LLOYAL_LOG_DEBUG(
        "[tokenizer::get_vocab] ERROR: llama_model_get_vocab returned NULL");
  }

  return vocab;
}

/**
 * Check if token is end-of-generation marker
 *
 * @param vocab Vocabulary from get_vocab()
 * @param token Token ID to check
 * @return True if token marks end of generation
 */
inline bool is_eog(const llama_vocab *vocab, llama_token token) {
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[tokenizer::is_eog] ERROR: vocab is null");
    return false;
  }

  return llama_vocab_is_eog(vocab, token);
}

/**
 * Get vocabulary size (total number of tokens)
 *
 * @param vocab Vocabulary from get_vocab()
 * @return Number of tokens in vocabulary
 */
inline int32_t vocab_size(const llama_vocab *vocab) {
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[tokenizer::vocab_size] ERROR: vocab is null");
    return 0;
  }

  return llama_vocab_n_tokens(vocab);
}

// ===== MODEL-ACCEPTING CONVENIENCE OVERLOADS =====
//
// These overloads accept llama_model* and handle vocab extraction + metadata
// queries internally. They delegate to the vocab-accepting primitives above.
//
// Benefits:
// - Eliminate boilerplate (vocab extraction, add_bos queries) in calling code
// - Reduce code duplication across projects
// - Backwards compatible - existing code unchanged

/**
 * Tokenize text to token array (model-accepting overload)
 *
 * Convenience wrapper that handles:
 * - Vocab extraction from model
 * - add_bos detection from GGUF metadata
 * - Special token parsing
 *
 * @param model Llama model
 * @param text Text to tokenize
 * @return Vector of token IDs
 */
inline std::vector<llama_token> tokenize(const llama_model *model,
                                         const std::string &text) {
  if (!model) {
    LLOYAL_LOG_DEBUG("[tokenizer::tokenize] ERROR: model is null");
    return {};
  }

  const llama_vocab *vocab = get_vocab(model);
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[tokenizer::tokenize] ERROR: get_vocab returned null");
    return {};
  }

  bool add_bos = llama_vocab_get_add_bos(vocab);
  return tokenize(vocab, text, add_bos, true);
}

/**
 * Detokenize SINGLE token to text (model-accepting overload)
 *
 * @param model Llama model
 * @param token Token ID to convert
 * @param special Enable special token rendering (default: true)
 * @return Text representation of token
 */
inline std::string detokenize(const llama_model *model, llama_token token,
                              bool special = true) {
  if (!model) {
    LLOYAL_LOG_DEBUG("[tokenizer::detokenize] ERROR: model is null");
    return "";
  }

  const llama_vocab *vocab = get_vocab(model);
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[tokenizer::detokenize] ERROR: get_vocab returned null");
    return "";
  }

  return detokenize(vocab, token, special);
}

/**
 * Detokenize TOKEN VECTOR to text (convenience overload)
 *
 * Accepts std::vector instead of raw pointer for safer API.
 *
 * @param model Llama model
 * @param tokens Vector of token IDs
 * @param remove_special Remove BOS/EOS tokens from output (default: false)
 * @param unparse_special Render special tokens as text (default: true)
 * @return Complete text representation
 */
inline std::string detokenize_batch(const llama_model *model,
                                    const std::vector<llama_token> &tokens,
                                    bool remove_special = false,
                                    bool unparse_special = true) {
  if (!model) {
    LLOYAL_LOG_DEBUG(
        "[tokenizer::detokenize_batch] ERROR: model is null");
    return "";
  }

  const llama_vocab *vocab = get_vocab(model);
  if (!vocab) {
    LLOYAL_LOG_DEBUG(
        "[tokenizer::detokenize_batch] ERROR: get_vocab returned null");
    return "";
  }

  return detokenize_batch(vocab, tokens.data(),
                          static_cast<int32_t>(tokens.size()), remove_special,
                          unparse_special);
}

/**
 * Detokenize TOKEN ARRAY to text (model-accepting overload)
 *
 * @param model Llama model
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens in array
 * @param remove_special Remove BOS/EOS tokens from output
 * @param unparse_special Render special tokens as text
 * @return Complete text representation
 */
inline std::string detokenize_batch(const llama_model *model,
                                    const llama_token *tokens, int32_t n_tokens,
                                    bool remove_special, bool unparse_special) {
  if (!model) {
    LLOYAL_LOG_DEBUG(
        "[tokenizer::detokenize_batch] ERROR: model is null");
    return "";
  }

  const llama_vocab *vocab = get_vocab(model);
  if (!vocab) {
    LLOYAL_LOG_DEBUG(
        "[tokenizer::detokenize_batch] ERROR: get_vocab returned null");
    return "";
  }

  return detokenize_batch(vocab, tokens, n_tokens, remove_special,
                          unparse_special);
}

/**
 * Check if token is end-of-generation marker (model-accepting overload)
 *
 * @param model Llama model
 * @param token Token ID to check
 * @return True if token marks end of generation
 */
inline bool is_eog(const llama_model *model, llama_token token) {
  if (!model) {
    LLOYAL_LOG_DEBUG("[tokenizer::is_eog] ERROR: model is null");
    return false;
  }

  const llama_vocab *vocab = get_vocab(model);
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[tokenizer::is_eog] ERROR: get_vocab returned null");
    return false;
  }

  return is_eog(vocab, token);
}

/**
 * Get vocabulary size (model-accepting overload)
 *
 * @param model Llama model
 * @return Number of tokens in vocabulary
 */
inline int32_t vocab_size(const llama_model *model) {
  if (!model) {
    LLOYAL_LOG_DEBUG("[tokenizer::vocab_size] ERROR: model is null");
    return 0;
  }

  const llama_vocab *vocab = get_vocab(model);
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[tokenizer::vocab_size] ERROR: get_vocab returned null");
    return 0;
  }

  return vocab_size(vocab);
}

} // namespace lloyal::tokenizer
