#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

#include "common.hpp"
#include "helpers.hpp" // For string_repeat, string_join, string_split
#include <lloyal/nlohmann/json.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * JSON Schema to Grammar Converter (Header-Only)
 *
 * Purpose: Convert JSON schema to GBNF (Grammar BNF) format for constrained
 * generation. Vendored from llama.cpp/common/json-schema-to-grammar.{h,cpp}
 *
 * Architecture:
 * - Public API: json_schema_to_grammar(), build_grammar()
 * - Internal: ~30 helper functions and SchemaConverter class in lloyal::detail
 * - Uses constant tables for primitive rules and format rules
 */

namespace lloyal {

using json = nlohmann::ordered_json;

// ===== PUBLIC API STRUCTS =====

struct common_grammar_builder {
  std::function<std::string(const std::string &, const std::string &)> add_rule;
  std::function<std::string(const std::string &, const json &)> add_schema;
  std::function<void(json &)> resolve_refs;
};

struct common_grammar_options {
  bool dotall = false;
};

// ===== PUBLIC API FUNCTIONS =====

/**
 * Convert JSON schema to GBNF grammar
 *
 * @param schema JSON schema (nlohmann::ordered_json)
 * @param force_gbnf Force GBNF output (default: false allows EBNF optimization)
 * @return GBNF grammar string
 */
std::string json_schema_to_grammar(const json &schema, bool force_gbnf = false);

/**
 * Build grammar from callback
 *
 * @param cb Callback function to build grammar rules
 * @param options Grammar options (dotall, etc.)
 * @return Formatted GBNF grammar string
 */
std::string
build_grammar(const std::function<void(const common_grammar_builder &)> &cb,
              const common_grammar_options &options = {});

} // namespace lloyal

namespace lloyal::detail {

// ===== CONSTANT TABLES =====

inline constexpr const char *SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
  std::string content;
  std::vector<std::string> deps;
};

/**
 * @var PRIMITIVE_RULES
 * @brief Built-in grammar rules for JSON primitives
 *
 * Defines GBNF rules for basic JSON types: boolean, number, integer, string, array,
 * object, null, uuid, and character escaping. Used as building blocks for schema conversion.
 */
inline const std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\") space", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number",
     {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? "
      "integral-part)? space",
      {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part) space", {"integral-part"}}},
    {"value",
     {"object | array | string | number | boolean | null",
      {"object", "array", "string", "number", "boolean", "null"}}},
    {"object",
     {"\"{\" space ( string \":\" space value (\",\" space string \":\" space "
      "value)* )? \"}\" space",
      {"string", "value"}}},
    {"array",
     {"\"[\" space ( value (\",\" space value)* )? \"]\" space", {"value"}}},
    {"uuid",
     {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" "
      "[0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\" space",
      {}}},
    {"char",
     {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" "
      "[0-9a-fA-F]{4})",
      {}}},
    {"string", {"\"\\\"\" char* \"\\\"\" space", {"char"}}},
    {"null", {"\"null\" space", {}}},
};

/**
 * @var STRING_FORMAT_RULES
 * @brief Grammar rules for string format validation
 *
 * Defines GBNF rules for JSON Schema string formats: date, time, date-time, uri, email, uuid.
 * Used when schema specifies "format" field for string validation.
 */
inline const std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES =
    {{"date",
      {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | "
       "[1-2] [0-9] | \"3\" [0-1] )",
       {}}},
     {"time",
      {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" "
       "[0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) "
       "\":\" [0-5] [0-9] )",
       {}}},
     {"date-time", {"date \"T\" time", {"date", "time"}}},
     {"date-string", {"\"\\\"\" date \"\\\"\" space", {"date"}}},
     {"time-string", {"\"\\\"\" time \"\\\"\" space", {"time"}}},
     {"date-time-string",
      {"\"\\\"\" date-time \"\\\"\" space", {"date-time"}}}};

/**
 * @brief Check if name conflicts with GBNF reserved keywords
 *
 * Tests whether a rule name would collide with built-in primitives ("root", "boolean",
 * "number", "string", etc.) or format rules ("date", "time", "uuid", etc.). Used during
 * schema conversion to append "-" suffix to conflicting names.
 *
 * @param name Rule name to check
 * @return True if name is reserved, false otherwise
 *
 * @note Uses lazy-initialized static set for O(1) lookup after first call
 */
inline bool is_reserved_name(const std::string &name) {
  static std::unordered_set<std::string> RESERVED_NAMES;
  if (RESERVED_NAMES.empty()) {
    RESERVED_NAMES.insert("root");
    for (const auto &p : PRIMITIVE_RULES)
      RESERVED_NAMES.insert(p.first);
    for (const auto &p : STRING_FORMAT_RULES)
      RESERVED_NAMES.insert(p.first);
  }
  return RESERVED_NAMES.find(name) != RESERVED_NAMES.end();
}

// Regex patterns for escaping
inline std::regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");
inline std::regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"]");
inline std::regex GRAMMAR_RANGE_LITERAL_ESCAPE_RE("[\r\n\"\\]\\-\\\\]");

inline const std::unordered_map<char, std::string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'-', "\\-"}, {']', "\\]"}};

inline const std::unordered_set<char> NON_LITERAL_SET = {
    '|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?'};
inline const std::unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {
    '^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

// ===== INTERNAL HELPER FUNCTIONS =====

inline std::string build_repetition(const std::string &item_rule, int min_items,
                                    int max_items,
                                    const std::string &separator_rule = "") {
  auto has_max = max_items != std::numeric_limits<int>::max();

  if (max_items == 0) {
    return "";
  }
  if (min_items == 0 && max_items == 1) {
    return item_rule + "?";
  }

  if (separator_rule.empty()) {
    if (min_items == 1 && !has_max) {
      return item_rule + "+";
    } else if (min_items == 0 && !has_max) {
      return item_rule + "*";
    } else {
      return item_rule + "{" + std::to_string(min_items) + "," +
             (has_max ? std::to_string(max_items) : "") + "}";
    }
  }

  auto result = item_rule + " " +
                build_repetition("(" + separator_rule + " " + item_rule + ")",
                                 min_items == 0 ? 0 : min_items - 1,
                                 has_max ? max_items - 1 : max_items);
  if (min_items == 0) {
    result = "(" + result + ")?";
  }
  return result;
}

inline void _build_min_max_int(int min_value, int max_value,
                               std::stringstream &out, int decimals_left = 16,
                               bool top_level = true) {
  auto has_min = min_value != std::numeric_limits<int>::min();
  auto has_max = max_value != std::numeric_limits<int>::max();

  auto digit_range = [&](char from, char to) {
    out << "[";
    if (from == to) {
      out << from;
    } else {
      out << from << "-" << to;
    }
    out << "]";
  };
  auto more_digits = [&](int min_digits, int max_digits) {
    out << "[0-9]";
    if (min_digits == max_digits && min_digits == 1) {
      return;
    }
    out << "{";
    out << min_digits;
    if (max_digits != min_digits) {
      out << ",";
      if (max_digits != std::numeric_limits<int>::max()) {
        out << max_digits;
      }
    }
    out << "}";
  };
  std::function<void(const std::string_view &, const std::string_view &)>
      uniform_range =
          [&](const std::string_view &from, const std::string_view &to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) {
              i++;
            }
            if (i > 0) {
              out << "\"" << from.substr(0, i) << "\"";
            }
            if (i < from.length() && i < to.length()) {
              if (i > 0) {
                out << " ";
              }
              auto sub_len = from.length() - i - 1;
              if (sub_len > 0) {
                auto from_sub = from.substr(i + 1);
                auto to_sub = to.substr(i + 1);
                auto sub_zeros = lloyal::string_repeat("0", sub_len);
                auto sub_nines = lloyal::string_repeat("9", sub_len);

                auto to_reached = false;
                out << "(";
                if (from_sub == sub_zeros) {
                  digit_range(from[i], to[i] - 1);
                  out << " ";
                  more_digits(sub_len, sub_len);
                } else {
                  out << "[" << from[i] << "] ";
                  out << "(";
                  uniform_range(from_sub, sub_nines);
                  out << ")";
                  if (from[i] < to[i] - 1) {
                    out << " | ";
                    if (to_sub == sub_nines) {
                      digit_range(from[i] + 1, to[i]);
                      to_reached = true;
                    } else {
                      digit_range(from[i] + 1, to[i] - 1);
                    }
                    out << " ";
                    more_digits(sub_len, sub_len);
                  }
                }
                if (!to_reached) {
                  out << " | ";
                  digit_range(to[i], to[i]);
                  out << " ";
                  uniform_range(sub_zeros, to_sub);
                }
                out << ")";
              } else {
                out << "[" << from[i] << "-" << to[i] << "]";
              }
            }
          };

  if (has_min && has_max) {
    if (min_value < 0 && max_value < 0) {
      out << "\"-\" (";
      _build_min_max_int(-max_value, -min_value, out, decimals_left,
                         /* top_level= */ true);
      out << ")";
      return;
    }

    if (min_value < 0) {
      out << "\"-\" (";
      _build_min_max_int(0, -min_value, out, decimals_left,
                         /* top_level= */ true);
      out << ") | ";
      min_value = 0;
    }

    auto min_s = std::to_string(min_value);
    auto max_s = std::to_string(max_value);
    auto min_digits = min_s.length();
    auto max_digits = max_s.length();

    for (auto digits = min_digits; digits < max_digits; digits++) {
      uniform_range(min_s, lloyal::string_repeat("9", digits));
      min_s = "1" + lloyal::string_repeat("0", digits);
      out << " | ";
    }
    uniform_range(min_s, max_s);
    return;
  }

  auto less_decimals = std::max(decimals_left - 1, 1);

  if (has_min) {
    if (min_value < 0) {
      out << "\"-\" (";
      _build_min_max_int(std::numeric_limits<int>::min(), -min_value, out,
                         decimals_left, /* top_level= */ false);
      out << ") | [0] | [1-9] ";
      more_digits(0, decimals_left - 1);
    } else if (min_value == 0) {
      if (top_level) {
        out << "[0] | [1-9] ";
        more_digits(0, less_decimals);
      } else {
        more_digits(1, decimals_left);
      }
    } else if (min_value <= 9) {
      char c = '0' + min_value;
      auto range_start = top_level ? '1' : '0';
      if (c > range_start) {
        digit_range(range_start, c - 1);
        out << " ";
        more_digits(1, less_decimals);
        out << " | ";
      }
      digit_range(c, '9');
      out << " ";
      more_digits(0, less_decimals);
    } else {
      auto min_s = std::to_string(min_value);
      auto len = min_s.length();
      auto c = min_s[0];

      if (c > '1') {
        digit_range(top_level ? '1' : '0', c - 1);
        out << " ";
        more_digits(len, less_decimals);
        out << " | ";
      }
      digit_range(c, c);
      out << " (";
      _build_min_max_int(std::stoi(min_s.substr(1)),
                         std::numeric_limits<int>::max(), out, less_decimals,
                         /* top_level= */ false);
      out << ")";
      if (c < '9') {
        out << " | ";
        digit_range(c + 1, '9');
        out << " ";
        more_digits(len - 1, less_decimals);
      }
    }
    return;
  }

  if (has_max) {
    if (max_value >= 0) {
      if (top_level) {
        out << "\"-\" [1-9] ";
        more_digits(0, less_decimals);
        out << " | ";
      }
      _build_min_max_int(0, max_value, out, decimals_left,
                         /* top_level= */ true);
    } else {
      out << "\"-\" (";
      _build_min_max_int(-max_value, std::numeric_limits<int>::max(), out,
                         decimals_left, /* top_level= */ false);
      out << ")";
    }
    return;
  }

  throw std::runtime_error(
      "At least one of min_value or max_value must be set");
}

inline std::string replacePattern(
    const std::string &input, const std::regex &regex,
    const std::function<std::string(const std::smatch &)> &replacement) {
  std::smatch match;
  std::string result;

  std::string::const_iterator searchStart(input.cbegin());
  std::string::const_iterator searchEnd(input.cend());

  while (std::regex_search(searchStart, searchEnd, match, regex)) {
    result.append(searchStart, searchStart + match.position());
    result.append(replacement(match));
    searchStart = match.suffix().first;
  }

  result.append(searchStart, searchEnd);

  return result;
}

inline std::string format_literal(const std::string &literal) {
  std::string escaped = replacePattern(literal, GRAMMAR_LITERAL_ESCAPE_RE,
                                       [&](const std::smatch &match) {
                                         char c = match.str()[0];
                                         return GRAMMAR_LITERAL_ESCAPES.at(c);
                                       });
  return "\"" + escaped + "\"";
}

// Forward declare SchemaConverter for build_grammar
class SchemaConverter;

} // namespace lloyal::detail

// Declare build_grammar here so SchemaConverter can be friended
namespace lloyal {
std::string
build_grammar(const std::function<void(const common_grammar_builder &)> &cb,
              const common_grammar_options &options);
}

namespace lloyal::detail {

// ===== SCHEMA CONVERTER CLASS =====

class SchemaConverter {
private:
  friend std::string lloyal::build_grammar(
      const std::function<void(const common_grammar_builder &)> &cb,
      const common_grammar_options &options);

  std::function<json(const std::string &)> _fetch_json;
  bool _dotall;
  std::map<std::string, std::string> _rules;
  std::unordered_map<std::string, json> _refs;
  std::unordered_set<std::string> _refs_being_resolved;
  std::vector<std::string> _errors;
  std::vector<std::string> _warnings;

  std::string _add_rule(const std::string &name, const std::string &rule);
  std::string _generate_union_rule(const std::string &name,
                                   const std::vector<json> &alt_schemas);
  std::string _visit_pattern(const std::string &pattern,
                             const std::string &name);
  std::string _not_strings(const std::vector<std::string> &strings);
  std::string _resolve_ref(const std::string &ref);
  std::string _build_object_rule(
      const std::vector<std::pair<std::string, json>> &properties,
      const std::unordered_set<std::string> &required, const std::string &name,
      const json &additional_properties);
  std::string _add_primitive(const std::string &name, const BuiltinRule &rule);

public:
  inline SchemaConverter(
      const std::function<json(const std::string &)> &fetch_json, bool dotall)
      : _fetch_json(fetch_json), _dotall(dotall) {
    _rules["space"] = SPACE_RULE;
  }

  /**
   * @brief Resolve $ref pointers in JSON schema
   *
   * Recursively resolves all $ref fields in schema, fetching remote schemas as needed.
   * Replaces relative refs (#/definitions/...) with absolute URLs and populates internal
   * _refs map with resolved schema objects.
   *
   * @param schema Schema object to resolve (modified in place)
   * @param url Base URL for resolving relative references
   *
   * @note Handles both absolute (https://...) and relative (#/definitions/...) refs
   * @note Errors accumulated in _errors vector for batch reporting
   */
  void resolve_refs(json &schema, const std::string &url);
  std::string _generate_constant_rule(const json &value);

  /**
   * @brief Convert schema node to GBNF rule
   *
   * Main entry point for schema-to-grammar conversion. Dispatches to appropriate handler
   * based on schema type (object, array, string, number, enum, etc.). Recursively processes
   * nested schemas and generates corresponding GBNF rules.
   *
   * @param schema Schema node to convert (JSON object, may contain type, properties, items, etc.)
   * @param name Rule name to generate (used as identifier in output grammar)
   * @return Generated GBNF rule definition
   *
   * @note Accumulates errors in _errors vector - call check_errors() after conversion
   * @note May throw std::runtime_error on unrecognized schema constructs
   *
   * @warning Complex method (~200+ lines) - handles all JSON Schema type keywords
   */
  std::string visit(const json &schema, const std::string &name);
  void check_errors();
  std::string format_grammar();
};

// Due to the complexity and length of the implementation, I'll include the key
// methods inline The full implementation follows the exact pattern from
// json-schema-to-grammar.cpp

inline std::string SchemaConverter::_add_rule(const std::string &name,
                                              const std::string &rule) {
  std::string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
  if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
    _rules[esc_name] = rule;
    return esc_name;
  } else {
    int i = 0;
    while (_rules.find(esc_name + std::to_string(i)) != _rules.end() &&
           _rules[esc_name + std::to_string(i)] != rule) {
      i++;
    }
    std::string key = esc_name + std::to_string(i);
    _rules[key] = rule;
    return key;
  }
}

inline std::string
SchemaConverter::_generate_union_rule(const std::string &name,
                                      const std::vector<json> &alt_schemas) {
  std::vector<std::string> rules;
  for (size_t i = 0; i < alt_schemas.size(); i++) {
    rules.push_back(
        visit(alt_schemas[i], name + (name.empty() ? "alternative-" : "-") +
                                  std::to_string(i)));
  }
  return lloyal::string_join(rules, " | ");
}

// The remaining methods follow the exact implementation from the source file...
// Due to length constraints, I'm including the essential structure.
// The full ~1000 line implementation should be copied from
// json-schema-to-grammar.cpp with the following conversions:
// 1. All static functions → inline functions in detail namespace
// 2. All member functions → inline member functions
// 3. string_repeat/join/split → lloyal::string_repeat/join/split
// 4. PRIMITIVE_RULES/STRING_FORMAT_RULES →
// detail::PRIMITIVE_RULES/STRING_FORMAT_RULES

// ===== MISSING METHOD IMPLEMENTATIONS =====

inline std::string SchemaConverter::_visit_pattern(const std::string &pattern,
                                                   const std::string &name) {
  if (!(pattern.front() == '^' && pattern.back() == '$')) {
    _errors.push_back("Pattern must start with '^' and end with '$'");
    return "";
  }
  std::string sub_pattern = pattern.substr(1, pattern.length() - 2);
  std::unordered_map<std::string, std::string> sub_rule_ids;

  size_t i = 0;
  size_t length = sub_pattern.length();

  using literal_or_rule = std::pair<std::string, bool>;
  auto to_rule = [&](const literal_or_rule &ls) {
    auto is_literal = ls.second;
    auto s = ls.first;
    return is_literal ? "\"" + s + "\"" : s;
  };
  std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
    size_t start = i;
    std::vector<literal_or_rule> seq;

    auto get_dot = [&]() {
      std::string rule;
      if (_dotall) {
        rule = "[\\U00000000-\\U0010FFFF]";
      } else {
        rule = "[^\\x0A\\x0D]";
      }
      return _add_rule("dot", rule);
    };

    // Joins the sequence, merging consecutive literals together.
    auto join_seq = [&]() {
      std::vector<literal_or_rule> ret;

      std::string literal;
      auto flush_literal = [&]() {
        if (literal.empty()) {
          return false;
        }
        ret.emplace_back(literal, true);
        literal.clear();
        return true;
      };

      for (const auto &item : seq) {
        auto is_literal = item.second;
        if (is_literal) {
          literal += item.first;
        } else {
          flush_literal();
          ret.push_back(item);
        }
      }
      flush_literal();

      std::vector<std::string> results;
      for (const auto &item : ret) {
        results.push_back(to_rule(item));
      }
      return std::make_pair(lloyal::string_join(results, " "), false);
    };

    while (i < length) {
      char c = sub_pattern[i];
      if (c == '.') {
        seq.emplace_back(get_dot(), false);
        i++;
      } else if (c == '(') {
        i++;
        if (i < length) {
          if (sub_pattern[i] == '?') {
            _warnings.push_back("Unsupported pattern syntax");
          }
        }
        seq.emplace_back("(" + to_rule(transform()) + ")", false);
      } else if (c == ')') {
        i++;
        if (start > 0 && sub_pattern[start - 1] != '(') {
          _errors.push_back("Unbalanced parentheses");
        }
        return join_seq();
      } else if (c == '[') {
        std::string square_brackets = std::string(1, c);
        i++;
        while (i < length && sub_pattern[i] != ']') {
          if (sub_pattern[i] == '\\') {
            square_brackets += sub_pattern.substr(i, 2);
            i += 2;
          } else {
            square_brackets += sub_pattern[i];
            i++;
          }
        }
        if (i >= length) {
          _errors.push_back("Unbalanced square brackets");
        }
        square_brackets += ']';
        i++;
        seq.emplace_back(square_brackets, false);
      } else if (c == '|') {
        seq.emplace_back("|", false);
        i++;
      } else if (c == '*' || c == '+' || c == '?') {
        seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
        i++;
      } else if (c == '{') {
        std::string curly_brackets = std::string(1, c);
        i++;
        while (i < length && sub_pattern[i] != '}') {
          curly_brackets += sub_pattern[i];
          i++;
        }
        if (i >= length) {
          _errors.push_back("Unbalanced curly brackets");
        }
        curly_brackets += '}';
        i++;
        auto nums = lloyal::string_split(
            curly_brackets.substr(1, curly_brackets.length() - 2), ",");
        int min_times = 0;
        int max_times = std::numeric_limits<int>::max();
        try {
          if (nums.size() == 1) {
            min_times = max_times = std::stoi(nums[0]);
          } else if (nums.size() != 2) {
            _errors.push_back("Wrong number of values in curly brackets");
          } else {
            if (!nums[0].empty()) {
              min_times = std::stoi(nums[0]);
            }
            if (!nums[1].empty()) {
              max_times = std::stoi(nums[1]);
            }
          }
        } catch (const std::invalid_argument &e) {
          _errors.push_back("Invalid number in curly brackets");
          return std::make_pair("", false);
        }
        auto &last = seq.back();
        auto &sub = last.first;
        auto sub_is_literal = last.second;

        if (!sub_is_literal) {
          std::string &sub_id = sub_rule_ids[sub];
          if (sub_id.empty()) {
            sub_id = _add_rule(name + "-" + std::to_string(sub_rule_ids.size()),
                               sub);
          }
          sub = sub_id;
        }
        seq.back().first = build_repetition(
            sub_is_literal ? "\"" + sub + "\"" : sub, min_times, max_times, "");
        seq.back().second = false;
      } else {
        std::string literal;
        auto is_non_literal = [&](char c) {
          return NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end();
        };
        while (i < length) {
          if (sub_pattern[i] == '\\' && i < length - 1) {
            char next = sub_pattern[i + 1];
            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.find(next) !=
                ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.end()) {
              i++;
              literal += sub_pattern[i];
              i++;
            } else {
              literal += sub_pattern.substr(i, 2);
              i += 2;
            }
          } else if (sub_pattern[i] == '"') {
            literal += "\\\"";
            i++;
          } else if (!is_non_literal(sub_pattern[i]) &&
                     (i == length - 1 || literal.empty() ||
                      sub_pattern[i + 1] == '.' ||
                      !is_non_literal(sub_pattern[i + 1]))) {
            literal += sub_pattern[i];
            i++;
          } else {
            break;
          }
        }
        if (!literal.empty()) {
          seq.emplace_back(literal, true);
        }
      }
    }
    return join_seq();
  };
  return _add_rule(name,
                   "\"\\\"\" (" + to_rule(transform()) + ") \"\\\"\" space");
}

inline std::string
SchemaConverter::_not_strings(const std::vector<std::string> &strings) {
  struct TrieNode {
    std::map<char, TrieNode> children;
    bool is_end_of_string;

    TrieNode() : is_end_of_string(false) {}

    void insert(const std::string &string) {
      auto node = this;
      for (char c : string) {
        node = &node->children[c];
      }
      node->is_end_of_string = true;
    }
  };

  TrieNode trie;
  for (const auto &s : strings) {
    trie.insert(s);
  }

  std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
  std::ostringstream out;
  out << "[\"] ( ";
  std::function<void(const TrieNode &)> visit = [&](const TrieNode &node) {
    std::ostringstream rejects;
    auto first = true;
    for (const auto &kv : node.children) {
      rejects << kv.first;
      if (first) {
        first = false;
      } else {
        out << " | ";
      }
      out << "[" << kv.first << "]";
      if (!kv.second.children.empty()) {
        out << " (";
        visit(kv.second);
        out << ")";
      } else if (kv.second.is_end_of_string) {
        out << " " << char_rule << "+";
      }
    }
    if (!node.children.empty()) {
      if (!first) {
        out << " | ";
      }
      out << "[^\"" << rejects.str() << "] " << char_rule << "*";
    }
  };
  visit(trie);

  out << " )";
  if (!trie.is_end_of_string) {
    out << "?";
  }
  out << " [\"] space";
  return out.str();
}

inline std::string SchemaConverter::_resolve_ref(const std::string &ref) {
  std::string ref_name = ref.substr(ref.find_last_of('/') + 1);
  if (_rules.find(ref_name) == _rules.end() &&
      _refs_being_resolved.find(ref) == _refs_being_resolved.end()) {
    _refs_being_resolved.insert(ref);
    json resolved = _refs[ref];
    ref_name = visit(resolved, ref_name);
    _refs_being_resolved.erase(ref);
  }
  return ref_name;
}

inline std::string SchemaConverter::_build_object_rule(
    const std::vector<std::pair<std::string, json>> &properties,
    const std::unordered_set<std::string> &required, const std::string &name,
    const json &additional_properties) {
  std::vector<std::string> required_props;
  std::vector<std::string> optional_props;
  std::unordered_map<std::string, std::string> prop_kv_rule_names;
  std::vector<std::string> prop_names;
  for (const auto &kv : properties) {
    const auto &prop_name = kv.first;
    const auto &prop_schema = kv.second;

    std::string prop_rule_name =
        visit(prop_schema, name + (name.empty() ? "" : "-") + prop_name);
    prop_kv_rule_names[prop_name] =
        _add_rule(name + (name.empty() ? "" : "-") + prop_name + "-kv",
                  format_literal(json(prop_name).dump()) +
                      " space \":\" space " + prop_rule_name);
    if (required.find(prop_name) != required.end()) {
      required_props.push_back(prop_name);
    } else {
      optional_props.push_back(prop_name);
    }
    prop_names.push_back(prop_name);
  }
  if ((additional_properties.is_boolean() &&
       additional_properties.get<bool>()) ||
      additional_properties.is_object()) {
    std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
    std::string value_rule =
        additional_properties.is_object()
            ? visit(additional_properties, sub_name + "-value")
            : _add_primitive("value", PRIMITIVE_RULES.at("value"));

    auto key_rule = prop_names.empty()
                        ? _add_primitive("string", PRIMITIVE_RULES.at("string"))
                        : _add_rule(sub_name + "-k", _not_strings(prop_names));
    std::string kv_rule =
        _add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
    prop_kv_rule_names["*"] = kv_rule;
    optional_props.push_back("*");
  }

  std::string rule = "\"{\" space ";
  for (size_t i = 0; i < required_props.size(); i++) {
    if (i > 0) {
      rule += " \",\" space ";
    }
    rule += prop_kv_rule_names[required_props[i]];
  }

  if (!optional_props.empty()) {
    rule += " (";
    if (!required_props.empty()) {
      rule += " \",\" space ( ";
    }

    std::function<std::string(const std::vector<std::string> &, bool)>
        get_recursive_refs = [&](const std::vector<std::string> &ks,
                                 bool first_is_optional) {
          std::string res;
          if (ks.empty()) {
            return res;
          }
          std::string k = ks[0];
          std::string kv_rule_name = prop_kv_rule_names[k];
          std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
          if (first_is_optional) {
            res = comma_ref + (k == "*" ? "*" : "?");
          } else {
            res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
          }
          if (ks.size() > 1) {
            res += " " +
                   _add_rule(name + (name.empty() ? "" : "-") + k + "-rest",
                             get_recursive_refs(std::vector<std::string>(
                                                    ks.begin() + 1, ks.end()),
                                                true));
          }
          return res;
        };

    for (size_t i = 0; i < optional_props.size(); i++) {
      if (i > 0) {
        rule += " | ";
      }
      rule += get_recursive_refs(
          std::vector<std::string>(optional_props.begin() + i,
                                   optional_props.end()),
          false);
    }
    if (!required_props.empty()) {
      rule += " )";
    }
    rule += " )?";
  }

  rule += " \"}\" space";

  return rule;
}

inline std::string SchemaConverter::_add_primitive(const std::string &name,
                                                   const BuiltinRule &rule) {
  auto n = _add_rule(name, rule.content);
  for (const auto &dep : rule.deps) {
    BuiltinRule dep_rule;
    auto it = PRIMITIVE_RULES.find(dep);
    if (it == PRIMITIVE_RULES.end()) {
      it = STRING_FORMAT_RULES.find(dep);
      if (it == STRING_FORMAT_RULES.end()) {
        _errors.push_back("Rule " + dep + " not known");
        continue;
      }
    }
    if (_rules.find(dep) == _rules.end()) {
      _add_primitive(dep, it->second);
    }
  }
  return n;
}

inline void SchemaConverter::resolve_refs(json &schema,
                                          const std::string &url) {
  /*
   * Resolves all $ref fields in the given schema, fetching any remote schemas,
   * replacing each $ref with absolute reference URL and populates _refs with
   * the respective referenced (sub)schema dictionaries.
   */
  std::function<void(json &)> visit_refs = [&](json &n) {
    if (n.is_array()) {
      for (auto &x : n) {
        visit_refs(x);
      }
    } else if (n.is_object()) {
      if (n.contains("$ref")) {
        std::string ref = n["$ref"];
        if (_refs.find(ref) == _refs.end()) {
          json target;
          if (ref.find("https://") == 0) {
            std::string base_url = ref.substr(0, ref.find('#'));
            auto it = _refs.find(base_url);
            if (it != _refs.end()) {
              target = it->second;
            } else {
              // Fetch the referenced schema and resolve its refs
              auto referenced = _fetch_json(ref);
              resolve_refs(referenced, base_url);
              _refs[base_url] = referenced;
            }
            if (ref.find('#') == std::string::npos ||
                ref.substr(ref.find('#') + 1).empty()) {
              return;
            }
          } else if (ref.find("#/") == 0) {
            target = schema;
            n["$ref"] = url + ref;
            ref = url + ref;
          } else {
            _errors.push_back("Unsupported ref: " + ref);
            return;
          }
          std::string pointer = ref.substr(ref.find('#') + 1);
          std::vector<std::string> tokens = lloyal::string_split(pointer, "/");
          for (size_t i = 1; i < tokens.size(); ++i) {
            std::string sel = tokens[i];
            if (target.is_null() || !target.contains(sel)) {
              _errors.push_back("Error resolving ref " + ref + ": " + sel +
                                " not in " + target.dump());
              return;
            }
            target = target[sel];
          }
          _refs[ref] = target;
        }
      } else {
        for (auto &kv : n.items()) {
          visit_refs(kv.value());
        }
      }
    }
  };

  visit_refs(schema);
}

inline std::string SchemaConverter::_generate_constant_rule(const json &value) {
  return format_literal(value.dump());
}

inline std::string SchemaConverter::visit(const json &schema,
                                          const std::string &name) {
  json schema_type = schema.contains("type") ? schema["type"] : json();
  std::string schema_format =
      schema.contains("format") ? schema["format"].get<std::string>() : "";
  std::string rule_name = is_reserved_name(name) ? name + "-"
                          : name.empty()         ? "root"
                                                 : name;

  if (schema.contains("$ref")) {
    return _add_rule(rule_name, _resolve_ref(schema["$ref"]));
  } else if (schema.contains("oneOf") || schema.contains("anyOf")) {
    std::vector<json> alt_schemas =
        schema.contains("oneOf") ? schema["oneOf"].get<std::vector<json>>()
                                 : schema["anyOf"].get<std::vector<json>>();
    return _add_rule(rule_name, _generate_union_rule(name, alt_schemas));
  } else if (schema_type.is_array()) {
    std::vector<json> schema_types;
    for (const auto &t : schema_type) {
      json schema_copy(schema);
      schema_copy["type"] = t;
      schema_types.push_back(schema_copy);
    }
    return _add_rule(rule_name, _generate_union_rule(name, schema_types));
  } else if (schema.contains("const")) {
    return _add_rule(rule_name,
                     _generate_constant_rule(schema["const"]) + " space");
  } else if (schema.contains("enum")) {
    std::vector<std::string> enum_values;
    for (const auto &v : schema["enum"]) {
      enum_values.push_back(_generate_constant_rule(v));
    }
    return _add_rule(rule_name,
                     "(" + lloyal::string_join(enum_values, " | ") + ") space");
  } else if ((schema_type.is_null() || schema_type == "object") &&
             (schema.contains("properties") ||
              (schema.contains("additionalProperties") &&
               schema["additionalProperties"] != true))) {
    std::unordered_set<std::string> required;
    if (schema.contains("required") && schema["required"].is_array()) {
      for (const auto &item : schema["required"]) {
        if (item.is_string()) {
          required.insert(item.get<std::string>());
        }
      }
    }
    std::vector<std::pair<std::string, json>> properties;
    if (schema.contains("properties")) {
      for (const auto &prop : schema["properties"].items()) {
        properties.emplace_back(prop.key(), prop.value());
      }
    }
    return _add_rule(rule_name,
                     _build_object_rule(properties, required, name,
                                        schema.contains("additionalProperties")
                                            ? schema["additionalProperties"]
                                            : json()));
  } else if ((schema_type.is_null() || schema_type == "object" ||
              schema_type == "string") &&
             schema.contains("allOf")) {
    std::unordered_set<std::string> required;
    std::vector<std::pair<std::string, json>> properties;
    std::map<std::string, size_t> enum_values;
    std::string hybrid_name = name;
    std::function<void(const json &, bool)> add_component =
        [&](const json &comp_schema, bool is_required) {
          if (comp_schema.contains("$ref")) {
            add_component(_refs[comp_schema["$ref"]], is_required);
          } else if (comp_schema.contains("properties")) {
            for (const auto &prop : comp_schema["properties"].items()) {
              properties.emplace_back(prop.key(), prop.value());
              if (is_required) {
                required.insert(prop.key());
              }
            }
          } else if (comp_schema.contains("enum")) {
            for (const auto &v : comp_schema["enum"]) {
              const auto rule = _generate_constant_rule(v);
              if (enum_values.find(rule) == enum_values.end()) {
                enum_values[rule] = 0;
              }
              enum_values[rule] += 1;
            }
          } else {
            // todo warning
          }
        };
    for (auto &t : schema["allOf"]) {
      if (t.contains("anyOf")) {
        for (auto &tt : t["anyOf"]) {
          add_component(tt, false);
        }
      } else {
        add_component(t, true);
      }
    }
    if (!enum_values.empty()) {
      std::vector<std::string> enum_intersection;
      for (const auto &p : enum_values) {
        if (p.second == schema["allOf"].size()) {
          enum_intersection.push_back(p.first);
        }
      }
      if (!enum_intersection.empty()) {
        return _add_rule(rule_name,
                         "(" + lloyal::string_join(enum_intersection, " | ") +
                             ") space");
      }
    }
    return _add_rule(rule_name, _build_object_rule(properties, required,
                                                   hybrid_name, json()));
  } else if ((schema_type.is_null() || schema_type == "array") &&
             (schema.contains("items") || schema.contains("prefixItems"))) {
    json items =
        schema.contains("items") ? schema["items"] : schema["prefixItems"];
    if (items.is_array()) {
      std::string rule = "\"[\" space ";
      for (size_t i = 0; i < items.size(); i++) {
        if (i > 0) {
          rule += " \",\" space ";
        }
        rule += visit(items[i], name + (name.empty() ? "" : "-") + "tuple-" +
                                    std::to_string(i));
      }
      rule += " \"]\" space";
      return _add_rule(rule_name, rule);
    } else {
      std::string item_rule_name =
          visit(items, name + (name.empty() ? "" : "-") + "item");
      int min_items =
          schema.contains("minItems") ? schema["minItems"].get<int>() : 0;
      json max_items_json =
          schema.contains("maxItems") ? schema["maxItems"] : json();
      int max_items = max_items_json.is_number_integer()
                          ? max_items_json.get<int>()
                          : std::numeric_limits<int>::max();

      return _add_rule(rule_name,
                       "\"[\" space " +
                           build_repetition(item_rule_name, min_items,
                                            max_items, "\",\" space") +
                           " \"]\" space");
    }
  } else if ((schema_type.is_null() || schema_type == "string") &&
             schema.contains("pattern")) {
    return _visit_pattern(schema["pattern"], rule_name);
  } else if ((schema_type.is_null() || schema_type == "string") &&
             std::regex_match(schema_format, std::regex("^uuid[1-5]?$"))) {
    return _add_primitive(rule_name == "root" ? "root" : schema_format,
                          PRIMITIVE_RULES.at("uuid"));
  } else if ((schema_type.is_null() || schema_type == "string") &&
             STRING_FORMAT_RULES.find(schema_format + "-string") !=
                 STRING_FORMAT_RULES.end()) {
    auto prim_name = schema_format + "-string";
    return _add_rule(
        rule_name,
        _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
  } else if (schema_type == "string" &&
             (schema.contains("minLength") || schema.contains("maxLength"))) {
    std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
    int min_len =
        schema.contains("minLength") ? schema["minLength"].get<int>() : 0;
    int max_len = schema.contains("maxLength")
                      ? schema["maxLength"].get<int>()
                      : std::numeric_limits<int>::max();
    return _add_rule(
        rule_name, "\"\\\"\" " + build_repetition(char_rule, min_len, max_len) +
                       " \"\\\"\" space");
  } else if (schema_type == "integer" &&
             (schema.contains("minimum") ||
              schema.contains("exclusiveMinimum") ||
              schema.contains("maximum") ||
              schema.contains("exclusiveMaximum"))) {
    int min_value = std::numeric_limits<int>::min();
    int max_value = std::numeric_limits<int>::max();
    if (schema.contains("minimum")) {
      min_value = schema["minimum"].get<int>();
    } else if (schema.contains("exclusiveMinimum")) {
      min_value = schema["exclusiveMinimum"].get<int>() + 1;
    }
    if (schema.contains("maximum")) {
      max_value = schema["maximum"].get<int>();
    } else if (schema.contains("exclusiveMaximum")) {
      max_value = schema["exclusiveMaximum"].get<int>() - 1;
    }
    std::stringstream out;
    out << "(";
    _build_min_max_int(min_value, max_value, out);
    out << ") space";
    return _add_rule(rule_name, out.str());
  } else if (schema.empty() || schema_type == "object") {
    return _add_rule(rule_name,
                     _add_primitive("object", PRIMITIVE_RULES.at("object")));
  } else {
    if (!schema_type.is_string() ||
        PRIMITIVE_RULES.find(schema_type.get<std::string>()) ==
            PRIMITIVE_RULES.end()) {
      _errors.push_back("Unrecognized schema: " + schema.dump());
      return "";
    }
    // TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at
    // least for zero
    return _add_primitive(rule_name == "root" ? "root"
                                              : schema_type.get<std::string>(),
                          PRIMITIVE_RULES.at(schema_type.get<std::string>()));
  }
}

inline void SchemaConverter::check_errors() {
  if (!_errors.empty()) {
    throw std::runtime_error("JSON schema conversion failed:\n" +
                             lloyal::string_join(_errors, "\n"));
  }
  if (!_warnings.empty()) {
    fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n",
            lloyal::string_join(_warnings, "; ").c_str());
  }
}

inline std::string SchemaConverter::format_grammar() {
  std::stringstream ss;
  for (const auto &kv : _rules) {
    ss << kv.first << " ::= " << kv.second << std::endl;
  }
  return ss.str();
}

} // namespace lloyal::detail

namespace lloyal {

// ===== PUBLIC API IMPLEMENTATION =====

inline std::string json_schema_to_grammar(const json &schema, bool force_gbnf) {
#ifdef LLAMA_USE_LLGUIDANCE
  if (!force_gbnf) {
    return "%llguidance {}\nstart: %json " + schema.dump();
  }
#else
  (void)force_gbnf;
#endif // LLAMA_USE_LLGUIDANCE
  return build_grammar([&](const common_grammar_builder &callbacks) {
    auto copy = schema;
    callbacks.resolve_refs(copy);
    callbacks.add_schema("", copy);
  });
}

inline std::string
build_grammar(const std::function<void(const common_grammar_builder &)> &cb,
              const common_grammar_options &options) {
  detail::SchemaConverter converter([&](const std::string &) { return json(); },
                                    options.dotall);
  common_grammar_builder builder{
      /* .add_rule = */ [&](const std::string &name, const std::string &rule) {
        return converter._add_rule(name, rule);
      },
      /* .add_schema = */
      [&](const std::string &name, const nlohmann::ordered_json &schema) {
        return converter.visit(schema, name == "root" ? "" : name);
      },
      /* .resolve_refs = */
      [&](nlohmann::ordered_json &schema) {
        converter.resolve_refs(schema, "");
      }};
  cb(builder);
  converter.check_errors();
  return converter.format_grammar();
}

} // namespace lloyal
