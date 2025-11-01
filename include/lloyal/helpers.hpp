#pragma once

#include "common.hpp"
#include <llama/llama.h>
#include <llama/ggml.h>
#include <vector>
#include <cassert>
#include <string>
#include <memory>
#include <chrono>
#include <sstream>

/**
 * Helper utilities vendored from llama.cpp/common/
 * MIT License - Copyright (c) 2023-2024 The ggml.ai team
 */

namespace lloyal {

// Forward declare for chat template
using json = nlohmann::ordered_json;

// ===== BATCH UTILITIES =====

// Clear batch to empty state (reset n_tokens)
inline void batch_clear(llama_batch& batch) {
    batch.n_tokens = 0;
}

// Add single token to batch with position and sequence info
inline void batch_add(llama_batch& batch,
                      llama_token id,
                      int32_t pos,
                      const std::vector<llama_seq_id>& seq_ids,
                      bool logits,
                      int32_t capacity = -1) {
    // Debug bounds checking to prevent buffer overflows
    #ifdef DEBUG
    if (capacity > 0) {
        assert(batch.n_tokens < capacity && "batch_add: token capacity exceeded");
    }
    #endif

    const auto i = batch.n_tokens;
    batch.token[i] = id;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = static_cast<int32_t>(seq_ids.size());
    for (size_t j = 0; j < seq_ids.size(); ++j) {
        batch.seq_id[i][j] = seq_ids[j];
    }
    batch.logits[i] = logits ? 1 : 0;
    batch.n_tokens++;
}

// ===== CHAT TEMPLATE TYPES (PUBLIC API) =====

struct ChatTemplateResult {
    std::string prompt;
    std::vector<std::string> additional_stops;
};

// Format chat messages using model's built-in template
inline std::string format_chat_template_from_model(
    const llama_model* model,
    const std::string& messages_json,
    const std::string& template_override = ""
) {
    try {
        json messages = json::parse(messages_json);

        // Determine template source
        std::string template_str;
        if (!template_override.empty()) {
            template_str = template_override;
        } else if (model) {
            const char* model_template = llama_model_chat_template(model, nullptr);
            if (model_template && strlen(model_template) > 0) {
                template_str = model_template;
            }
        }

        if (template_str.empty()) {
            template_str = detail::get_chatml_template();
        }

        // Get BOS/EOS tokens from model
        std::string bos_token, eos_token;
        if (model) {
            const auto* vocab = llama_model_get_vocab(model);
            bos_token = detail::get_token_safe(model, llama_vocab_bos(vocab));
            eos_token = detail::get_token_safe(model, llama_vocab_eos(vocab));
        }

        return detail::apply_chat_template_helper(template_str, messages, bos_token, eos_token, true);

    } catch (const std::exception& e) {
        return "";
    }
}

// Dynamic stop token detection
inline std::vector<std::string> extract_template_stop_tokens(
    const llama_model* model,
    const std::string& template_str
) {
    std::vector<std::string> stops;

    if (!model) return stops;

    const auto* vocab = llama_model_get_vocab(model);
    if (!vocab) return stops;

    // Check what tokens actually exist in this model's vocabulary
    const auto get_token_if_exists = [&](const std::string& token_text) -> std::string {
        std::vector<llama_token> tokens(1);
        int n_tokens = llama_tokenize(vocab, token_text.c_str(),
                                     static_cast<int32_t>(token_text.length()),
                                     tokens.data(), 1, false, true);
        if (n_tokens == 1) {
            return token_text;
        }
        return "";
    };

    // For ChatML-style templates
    if (template_str.find("im_start") != std::string::npos) {
        auto token = get_token_if_exists("<|im_end|>");
        if (!token.empty()) stops.push_back(token);

        token = get_token_if_exists("<|endoftext|>");
        if (!token.empty()) stops.push_back(token);
    }

    // For Llama-3 style templates
    if (template_str.find("eom_id") != std::string::npos || template_str.find("eot_id") != std::string::npos) {
        auto token = get_token_if_exists("<|eom_id|>");
        if (!token.empty()) stops.push_back(token);

        token = get_token_if_exists("<|eot_id|>");
        if (!token.empty()) stops.push_back(token);
    }

    // Always check for model's EOT token as fallback
    auto eot_token = llama_vocab_eot(vocab);
    if (eot_token != LLAMA_TOKEN_NULL) {
        std::string eot_text = detail::common_token_to_piece(vocab, eot_token, true);
        if (!eot_text.empty() && std::find(stops.begin(), stops.end(), eot_text) == stops.end()) {
            stops.push_back(eot_text);
        }
    }

    return stops;
}

// Complete chat template processing
inline ChatTemplateResult format_chat_template_complete(
    const llama_model* model,
    const std::string& messages_json,
    const std::string& template_override = ""
) {
    ChatTemplateResult result;

    try {
        json messages = json::parse(messages_json);

        std::string template_str;
        if (!template_override.empty()) {
            template_str = template_override;
        } else if (model) {
            const char* model_template = llama_model_chat_template(model, nullptr);
            if (model_template && strlen(model_template) > 0) {
                template_str = model_template;
            }
        }

        if (template_str.empty()) {
            template_str = detail::get_chatml_template();
        }

        std::string bos_token, eos_token;
        if (model) {
            const auto* vocab = llama_model_get_vocab(model);
            bos_token = detail::get_token_safe(model, llama_vocab_bos(vocab));
            eos_token = detail::get_token_safe(model, llama_vocab_eos(vocab));
        }

        result.prompt = detail::apply_chat_template_helper(template_str, messages, bos_token, eos_token, true);
        result.additional_stops = extract_template_stop_tokens(model, template_str);

    } catch (const std::exception& e) {
        result.prompt = "";
        result.additional_stops.clear();
    }

    return result;
}

// Validate chat template syntax
inline bool validate_chat_template_helper(const std::string& template_str) {
    try {
        minja::chat_template tmpl(template_str, "", "");
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

// ===== PARAMETER CONVERSION HELPERS =====

// Get supported KV cache types
inline const std::vector<ggml_type>& get_kv_cache_types() {
    static const std::vector<ggml_type> types = {
        GGML_TYPE_F32,
        GGML_TYPE_F16,
        GGML_TYPE_BF16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
    };
    return types;
}

// Convert cache type string to ggml_type enum
inline ggml_type kv_cache_type_from_str(const std::string& s) {
    const auto& kv_cache_types = get_kv_cache_types();
    for (const auto& type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

// String validation helpers
inline bool is_truthy(const std::string& value) {
    return value == "on" || value == "enabled" || value == "1" || value == "true";
}

inline bool is_falsey(const std::string& value) {
    return value == "off" || value == "disabled" || value == "0" || value == "false";
}

inline bool is_autoy(const std::string& value) {
    return value == "auto" || value == "-1";
}

// ===== STRING UTILITIES =====

// Repeat string n times
inline std::string string_repeat(const std::string& str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

// Join strings with separator
inline std::string string_join(const std::vector<std::string>& values, const std::string& separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

// Split string by delimiter
inline std::vector<std::string> string_split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

} // namespace lloyal

namespace lloyal::detail {

// ===== INTERNAL TOKEN HELPERS =====

// Token conversion helper
inline std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        assert(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }
    return piece;
}

// Extract token from vocabulary safely
inline std::string get_token_safe(const llama_model* model, llama_token token) {
    if (!model || token == LLAMA_TOKEN_NULL) {
        return "";
    }
    const auto* vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, /* special */ true);
}

// ===== INTERNAL TEMPLATE HELPERS =====

// Default ChatML template fallback
inline const char* get_chatml_template() {
    return "{% for message in messages %}"
           "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
           "{% endfor %}"
           "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}";
}

// Apply chat template using minja engine (requires minja.hpp to be included first)
inline std::string apply_chat_template_helper(
    const std::string& template_str,
    const json& messages,
    const std::string& bos_token = "",
    const std::string& eos_token = "",
    bool add_generation_prompt = true
) {
    try {
        // Create minja chat template with correct 3-parameter constructor
        minja::chat_template tmpl(template_str, bos_token, eos_token);

        // Prepare template inputs
        minja::chat_template_inputs inputs;
        inputs.messages = messages;
        inputs.tools = json::array(); // No tools for basic implementation
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.now = std::chrono::system_clock::now();

        // Apply template with default options
        minja::chat_template_options opts;
        auto result = tmpl.apply(inputs, opts);

        // Handle BOS/EOS token removal
        if (!bos_token.empty() && result.starts_with(bos_token)) {
            result = result.substr(bos_token.length());
        }
        if (!eos_token.empty() && result.ends_with(eos_token)) {
            result = result.substr(0, result.length() - eos_token.length());
        }

        return result;
    } catch (const std::exception& e) {
        return "";
    }
}

} // namespace lloyal::detail
