// Stub for llama.cpp common/json-schema-to-grammar.h
// Provides JSON schema to GBNF conversion used by grammar.hpp

#pragma once

#include <nlohmann/json.hpp>
#include <string>

// Stub implementation - returns a simple grammar
inline std::string json_schema_to_grammar(
    const nlohmann::ordered_json& schema,
    bool force_gbnf
) {
  (void)schema;
  (void)force_gbnf;

  // Return a minimal valid GBNF grammar
  return R"(root ::= "{" "}"
)";
}
