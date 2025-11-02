#pragma once

#include "common.hpp"
#include "json-schema-to-grammar.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

/**
 * Grammar Anti-Corruption Layer (Header-Only)
 *
 * PURPOSE: Provides JSON schema to GBNF grammar conversion for structured
 * output
 *
 * ARCHITECTURE:
 * - This layer CALLS json_schema_to_grammar from json-schema-to-grammar.hpp
 * - Does NOT reimplement conversion logic
 * - Provides error handling, logging, and consistent API
 *
 * USAGE:
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

    // Call lloyal::json_schema_to_grammar from json-schema-to-grammar.hpp
    // Parameters: (schema, force_gbnf)
    // force_gbnf=false allows EBNF optimization when possible
    std::string grammar = lloyal::json_schema_to_grammar(schema, false);

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

} // namespace lloyal::grammar
