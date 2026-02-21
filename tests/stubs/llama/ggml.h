#pragma once

// Minimal ggml.h stub for testing
// Only includes types/functions used by helpers.hpp

#include <string>

// Quantization types used for KV cache
enum ggml_type {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_COUNT,  // Sentinel — must be last
};

// Stub implementation for type name conversion
inline std::string ggml_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_F16: return "f16";
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q4_1: return "q4_1";
        case GGML_TYPE_IQ4_NL: return "iq4_nl";
        case GGML_TYPE_Q5_0: return "q5_0";
        case GGML_TYPE_Q5_1: return "q5_1";
        default: return "unknown";
    }
}
