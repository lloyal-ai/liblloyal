#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <set>

// Minimal llama.cpp API surface for testing
// These types and functions are stubs - they mimic llama.cpp's interface
// without requiring actual llama.cpp linking

// Forward declarations matching llama.cpp types
struct llama_model {};
struct llama_context {};
struct llama_memory_t {};  // KV cache memory handle
struct llama_vocab {};      // Vocabulary handle
struct llama_sampler {};    // Sampler chain (new API commit b6451+)

// Type aliases matching llama.cpp
using llama_seq_id = int32_t;
using llama_pos = int32_t;
using llama_token = int32_t;

// Special token constants
#define LLAMA_TOKEN_NULL -1

// Batch structure matching llama.cpp
struct llama_batch {
    int32_t n_tokens;
    llama_token* token;
    int32_t* pos;
    int32_t* n_seq_id;
    llama_seq_id** seq_id;
    int8_t* logits;

    // Stub-specific: track allocated capacity for cleanup
    int32_t _capacity;
};

// Model parameters structure matching llama.cpp
struct llama_model_params {
    int n_gpu_layers = -1;  // Number of layers to offload to GPU (-1 = all)
    bool use_mmap = true;   // Use memory mapping for model loading
};

// Context parameters structure matching llama.cpp
struct llama_context_params {
    uint32_t n_ctx = 512;      // Context size
    uint32_t n_batch = 512;    // Batch size
};

// Sampler chain parameters (new API commit b6451+)
struct llama_sampler_chain_params {
    bool no_perf = true;       // Don't collect performance stats
};

// Token data structure for sampling (used by grammar sampler)
struct llama_token_data {
    llama_token id;  // Token ID
    float logit;     // Log-probability
    float p;         // Probability after softmax
};

// Token data array structure for sampling
struct llama_token_data_array {
    llama_token_data* data;
    size_t size;
    int64_t selected;  // Index of selected token (-1 = none)
    bool sorted;       // Whether array is sorted by probability
};

// Minimal llama.cpp API functions
extern "C" {
    // Model loading and cleanup
    llama_model* llama_model_load_from_file(const char* path, llama_model_params params);
    void llama_model_free(llama_model* model);
    size_t llama_model_size(const llama_model* model);

    // Context creation and cleanup
    llama_context* llama_init_from_model(llama_model* model, llama_context_params params);
    void llama_free(llama_context* ctx);

    // Default parameters
    llama_model_params llama_model_default_params();
    llama_context_params llama_context_default_params();

    // KV cache memory operations
    llama_memory_t llama_get_memory(llama_context* ctx);
    bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq, llama_pos p0, llama_pos p1);
    llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq);

    // Per-sequence state operations
    size_t llama_state_seq_get_size(llama_context* ctx, llama_seq_id seq);
    size_t llama_state_seq_get_data(llama_context* ctx, uint8_t* dst, size_t size, llama_seq_id seq);
    size_t llama_state_seq_set_data(llama_context* ctx, const uint8_t* src, size_t size, llama_seq_id seq);

    // Global state operations (fallback)
    size_t llama_state_get_size(llama_context* ctx);
    size_t llama_state_get_data(llama_context* ctx, uint8_t* dst, size_t size);
    size_t llama_state_set_data(llama_context* ctx, const uint8_t* src, size_t size);

    // Batch operations
    llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
    void llama_batch_free(llama_batch batch);

    // Decode operations
    int llama_decode(llama_context* ctx, llama_batch batch);

    // Tokenization operations
    int llama_tokenize(
        const llama_vocab* vocab,
        const char* text,
        int32_t text_len,
        llama_token* tokens,
        int32_t n_tokens_max,
        bool add_special,
        bool parse_special
    );

    int llama_detokenize(
        const llama_vocab* vocab,
        const llama_token* tokens,
        int32_t n_tokens,
        char* text,
        int32_t text_len,
        bool remove_special,
        bool unparse_special
    );

    int llama_token_to_piece(
        const llama_vocab* vocab,
        llama_token token,
        char* buf,
        int32_t length,
        int32_t lstrip,
        bool special
    );

    // Sampling operations
    float* llama_get_logits_ith(llama_context* ctx, int32_t i);
    int llama_vocab_n_tokens(const llama_vocab* vocab);
    bool llama_vocab_is_eog(const llama_vocab* vocab, llama_token token);
    const llama_vocab* llama_model_get_vocab(const llama_model* model);

    // Chat template and special tokens
    const char* llama_model_chat_template(const llama_model* model, const char* name);
    llama_token llama_vocab_bos(const llama_vocab* vocab);
    llama_token llama_vocab_eos(const llama_vocab* vocab);
    llama_token llama_vocab_eot(const llama_vocab* vocab);

    // Sampler operations (new API commit b6451+)
    llama_sampler_chain_params llama_sampler_chain_default_params();
    llama_sampler* llama_sampler_chain_init(llama_sampler_chain_params params);
    void llama_sampler_chain_add(llama_sampler* chain, llama_sampler* smpl);
    llama_sampler* llama_sampler_init_greedy();
    llama_sampler* llama_sampler_init_temp(float temp);
    llama_sampler* llama_sampler_init_dist(uint32_t seed);
    llama_sampler* llama_sampler_init_top_k(int32_t k);
    llama_sampler* llama_sampler_init_top_p(float p, size_t min_keep);
    llama_sampler* llama_sampler_init_min_p(float p, size_t min_keep);
    llama_sampler* llama_sampler_init_typical(float p, size_t min_keep);
    llama_sampler* llama_sampler_init_penalties(
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present
    );
    llama_token llama_sampler_sample(llama_sampler* smpl, llama_context* ctx, int32_t idx);
    void llama_sampler_free(llama_sampler* smpl);

    // Grammar sampler operations (for vendored common_sampler)
    llama_sampler* llama_sampler_init_grammar(const llama_vocab* vocab, const char* grammar_str, const char* grammar_root);
    void llama_sampler_apply(llama_sampler* smpl, llama_token_data_array* cur_p);
    void llama_sampler_accept(llama_sampler* smpl, llama_token token);
    void llama_sampler_reset(llama_sampler* smpl);

    // Model introspection (for vendored common_sampler)
    const llama_model* llama_get_model(const llama_context* ctx);
}

// Test control structure for configuring stub behavior
struct LlamaStubConfig {
    // Model operations
    bool model_load_succeeds = true;           // Controls if llama_model_load_from_file returns valid pointer
    size_t model_size = 4ULL * 1024 * 1024 * 1024;  // Default: 4GB model

    // Context operations
    bool context_init_succeeds = true;         // Controls if llama_init_from_model returns valid pointer

    // KV cache operations
    llama_pos pos_max = -1;                    // Max position in KV cache (-1 = empty)
    bool rm_ok = true;                         // Whether llama_memory_seq_rm succeeds

    // Per-sequence state operations
    size_t per_seq_size = 0;                   // Per-sequence state size (0 = triggers fallback)
    size_t per_seq_rw = 0;                     // Per-sequence read/write bytes (0 = triggers fallback)

    // Global state operations (fallback)
    size_t global_size = 0;                    // Global state size
    size_t global_rw = 0;                      // Global read/write bytes

    // Batch/Decode operations
    bool batch_init_succeeds = true;           // Controls if llama_batch_init succeeds
    int decode_result = 0;                     // 0=success, <0=failure
    int decode_call_count = 0;                 // Track number of decode calls
    int batch_free_call_count = 0;             // Track RAII cleanup

    // Tokenization operations (two-pass simulation)
    std::vector<llama_token> tokenize_result;  // Tokens to return
    bool tokenize_succeeds = true;             // Controls if tokenization succeeds

    // Detokenization operations (two-pass simulation)
    std::string detokenize_result;             // Text to return for batch detokenization
    bool detokenize_succeeds = true;           // Controls if detokenization succeeds

    // token_to_piece (two-pass simulation for single token)
    std::string token_piece;                   // Piece for single token
    bool token_piece_succeeds = true;          // Controls if token_to_piece succeeds

    // Sampling operations
    std::vector<float> logits;                 // Logits array for greedy sampling
    int vocab_size_value = 0;                  // Vocabulary size
    std::set<llama_token> eog_tokens;          // End-of-generation tokens
    llama_token sample_result = 1;             // Token returned by llama_sampler_sample

    // Chat template and special tokens
    std::string chat_template;                 // Model's chat template string
    llama_token bos_token = 1;                 // Beginning-of-sequence token
    llama_token eos_token = 2;                 // End-of-sequence token
    llama_token eot_token = 3;                 // End-of-turn token
};

// Global stub configuration accessor
LlamaStubConfig& llamaStubConfig();

// Reset stub to default configuration (call before each test)
void resetStubConfig();
