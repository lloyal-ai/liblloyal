#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

#include "common.hpp"
#include "tokenizer.hpp"
#include <llama/llama.h>
#include <json-schema-to-grammar.h>  // llama.cpp common library
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file grammar.hpp
 * @brief Grammar-Constrained Sampling
 *
 * Provides JSON schema to GBNF grammar conversion for structured output generation.
 * Uses json_schema_to_grammar() from llama.cpp's common library.
 *
 * Architecture:
 * - Calls json_schema_to_grammar() from common/json-schema-to-grammar.h
 * - Adds error handling, logging, and consistent API
 * - Manages grammar sampler lifecycle
 *
 * @example
 *   std::string gbnf = lloyal::grammar::from_json_schema(schemaJsonString);
 *   // Pass to sampler::sample_with_params() via grammarSampler parameter
 */

namespace lloyal::grammar {

/**
 * Convert JSON schema to GBNF (Grammar BNF) format
 *
 * @param schema_json JSON schema string (e.g., {"type": "object", "properties":
 * {...}})
 * @return GBNF grammar string compatible with llama_sampler_init_grammar()
 * @throws std::runtime_error on parse error or conversion failure
 *
 * EXAMPLE:
 *   std::string schema = R"({"type": "object", "properties": {"name": {"type":
 * "string"}}})"; std::string gbnf = grammar::from_json_schema(schema);
 */
inline std::string from_json_schema(const std::string &schema_json) {
  LLOYAL_LOG_DEBUG(
      "[grammar::from_json_schema] Converting JSON schema (%zu bytes)",
      schema_json.size());

  try {
    // Parse JSON schema
    nlohmann::ordered_json schema = nlohmann::ordered_json::parse(schema_json);

    LLOYAL_LOG_DEBUG("[grammar::from_json_schema] Schema parsed, calling "
                     "json_schema_to_grammar");

    // Call json_schema_to_grammar from llama.cpp common library
    // Parameters: (schema, force_gbnf)
    // force_gbnf=false allows EBNF optimization when possible
    std::string grammar = json_schema_to_grammar(schema, false);

    if (grammar.empty()) {
      LLOYAL_LOG_DEBUG("[grammar::from_json_schema] ERROR: Conversion produced "
                       "empty grammar");
      throw std::runtime_error("Grammar conversion produced empty result");
    }

    LLOYAL_LOG_DEBUG(
        "[grammar::from_json_schema] Generated GBNF grammar (%zu bytes)",
        grammar.size());
    return grammar;

  } catch (const nlohmann::json::parse_error &e) {
    std::string errorMsg = std::string("JSON parse error: ") + e.what();
    LLOYAL_LOG_DEBUG("[grammar::from_json_schema] ERROR: %s", errorMsg.c_str());
    throw std::runtime_error(errorMsg);
  } catch (const std::exception &e) {
    std::string errorMsg =
        std::string("Grammar conversion failed: ") + e.what();
    LLOYAL_LOG_DEBUG("[grammar::from_json_schema] ERROR: %s", errorMsg.c_str());
    throw std::runtime_error(errorMsg);
  }
}

// ===== SAMPLER INITIALIZATION =====

/**
 * Initialize a grammar sampler from GBNF grammar string
 *
 * Convenience wrapper that handles vocab extraction from model.
 *
 * @param model Llama model (for vocab extraction)
 * @param grammar_str GBNF grammar string (from from_json_schema or hand-written)
 * @param root_rule Root rule name (default: "root")
 * @return Initialized grammar sampler, or nullptr on failure
 *
 * OWNERSHIP: Caller owns returned sampler and must call llama_sampler_free()
 *
 * EXAMPLE:
 *   std::string gbnf = grammar::from_json_schema(schema);
 *   llama_sampler* sampler = grammar::init_sampler(model, gbnf);
 *   // ... use sampler ...
 *   llama_sampler_free(sampler);
 */
inline llama_sampler *init_sampler(const llama_model *model,
                                   const std::string &grammar_str,
                                   const std::string &root_rule = "root") {
  if (!model) {
    LLOYAL_LOG_DEBUG("[grammar::init_sampler] ERROR: model is null");
    return nullptr;
  }

  if (grammar_str.empty()) {
    LLOYAL_LOG_DEBUG("[grammar::init_sampler] ERROR: grammar_str is empty");
    return nullptr;
  }

  const llama_vocab *vocab = tokenizer::get_vocab(model);
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[grammar::init_sampler] ERROR: get_vocab returned null");
    return nullptr;
  }

  LLOYAL_LOG_DEBUG("[grammar::init_sampler] Initializing grammar sampler "
                   "(grammar: %zu bytes, root: %s)",
                   grammar_str.size(), root_rule.c_str());

  llama_sampler *sampler =
      llama_sampler_init_grammar(vocab, grammar_str.c_str(), root_rule.c_str());

  if (!sampler) {
    LLOYAL_LOG_DEBUG("[grammar::init_sampler] ERROR: "
                     "llama_sampler_init_grammar returned null");
  }

  return sampler;
}

/**
 * Initialize a lazy grammar sampler from GBNF grammar string
 *
 * Generation runs unconstrained until a trigger pattern or token fires,
 * at which point the grammar activates and constrains subsequent tokens.
 * Used for tool-call generation: model writes freely until `<tool_call>`,
 * then grammar forces valid XML structure.
 *
 * @param model Llama model (for vocab extraction)
 * @param grammar_str GBNF grammar string
 * @param trigger_patterns Regex patterns that activate the grammar
 * @param trigger_tokens Token IDs that activate the grammar
 * @param root_rule Root rule name (default: "root")
 * @return Initialized lazy grammar sampler, or nullptr on failure
 *
 * OWNERSHIP: Caller owns returned sampler and must call llama_sampler_free()
 */
inline llama_sampler *init_lazy_sampler(
    const llama_model *model,
    const std::string &grammar_str,
    const std::vector<std::string> &trigger_patterns,
    const std::vector<llama_token> &trigger_tokens,
    const std::string &root_rule = "root") {
  if (!model) {
    LLOYAL_LOG_DEBUG("[grammar::init_lazy_sampler] ERROR: model is null");
    return nullptr;
  }

  if (grammar_str.empty()) {
    LLOYAL_LOG_DEBUG("[grammar::init_lazy_sampler] ERROR: grammar_str is empty");
    return nullptr;
  }

  const llama_vocab *vocab = tokenizer::get_vocab(model);
  if (!vocab) {
    LLOYAL_LOG_DEBUG("[grammar::init_lazy_sampler] ERROR: get_vocab returned null");
    return nullptr;
  }

  std::vector<const char *> patterns_c;
  patterns_c.reserve(trigger_patterns.size());
  for (const auto &p : trigger_patterns) patterns_c.push_back(p.c_str());

  LLOYAL_LOG_DEBUG("[grammar::init_lazy_sampler] Initializing lazy grammar "
                   "(grammar: %zu bytes, %zu patterns, %zu tokens, root: %s)",
                   grammar_str.size(), trigger_patterns.size(),
                   trigger_tokens.size(), root_rule.c_str());

  llama_sampler *sampler = llama_sampler_init_grammar_lazy_patterns(
      vocab, grammar_str.c_str(), root_rule.c_str(),
      patterns_c.data(), patterns_c.size(),
      trigger_tokens.data(), trigger_tokens.size());

  if (!sampler) {
    LLOYAL_LOG_DEBUG("[grammar::init_lazy_sampler] ERROR: "
                     "llama_sampler_init_grammar_lazy_patterns returned null");
  }

  return sampler;
}

/**
 * Clone a grammar sampler (for fork/branching).
 *
 * Creates a deep copy of the sampler including its parser state.
 * Use when forking a stepper to preserve grammar position.
 *
 * @param smpl Source sampler to clone
 * @return New sampler with same state, or nullptr if input was null
 *
 * OWNERSHIP: Caller owns returned sampler and must call llama_sampler_free()
 */
inline llama_sampler *clone_sampler(llama_sampler *smpl) {
  if (!smpl) {
    LLOYAL_LOG_DEBUG("[grammar::clone_sampler] Input is null, returning null");
    return nullptr;
  }

  llama_sampler *cloned = llama_sampler_clone(smpl);

  if (!cloned) {
    LLOYAL_LOG_DEBUG("[grammar::clone_sampler] ERROR: llama_sampler_clone failed");
  } else {
    LLOYAL_LOG_DEBUG("[grammar::clone_sampler] Cloned sampler successfully");
  }

  return cloned;
}

/**
 * Free a grammar sampler
 *
 * @param smpl Sampler to free (safe to call with nullptr)
 */
inline void free_sampler(llama_sampler* smpl) {
  if (smpl) {
    llama_sampler_free(smpl);
  }
}

/**
 * Apply grammar constraint to candidates
 *
 * Modifies candidates in-place, masking tokens that violate grammar.
 *
 * @param smpl Grammar sampler
 * @param cur_p Candidate array (modified in-place)
 */
inline void apply(llama_sampler* smpl, llama_token_data_array* cur_p) {
  if (smpl && cur_p) {
    llama_sampler_apply(smpl, cur_p);
  }
}

/**
 * Accept a token into grammar state
 *
 * Advances the grammar parser state.
 *
 * @param smpl Grammar sampler
 * @param token Token to accept
 */
inline void accept(llama_sampler* smpl, llama_token token) {
  if (smpl) {
    llama_sampler_accept(smpl, token);
  }
}

} // namespace lloyal::grammar
