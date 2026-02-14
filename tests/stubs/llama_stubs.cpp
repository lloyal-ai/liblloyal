#include "llama_stubs.h"
#include <atomic>

// Global configuration instance
static LlamaStubConfig g_stub_config;

// Instance counters for generating unique fake pointers
static std::atomic<uintptr_t> g_model_counter{1};
static std::atomic<uintptr_t> g_context_counter{
    1000}; // Start at different offset

// Singleton memory handle (shared across all contexts for simplicity)
static llama_memory_t g_memory_handle;

// Accessor for global stub configuration
LlamaStubConfig &llamaStubConfig() { return g_stub_config; }

// Reset stub to default configuration
void resetStubConfig() { g_stub_config = LlamaStubConfig{}; }

// Stub implementations
extern "C" {

// ===== MODEL OPERATIONS =====

llama_model *llama_model_load_from_file(const char * /*path*/,
                                        llama_model_params /*params*/) {
  if (!g_stub_config.model_load_succeeds) {
    return nullptr;
  }

  // Generate unique fake pointer for this model instance
  uintptr_t fake_addr = g_model_counter.fetch_add(1);
  return reinterpret_cast<llama_model *>(fake_addr);
}

void llama_model_free(llama_model * /*model*/) {
  // Stub does nothing - no actual cleanup needed
}

size_t llama_model_size(const llama_model * /*model*/) {
  return g_stub_config.model_size;
}

llama_model_params llama_model_default_params() {
  llama_model_params params;
  params.n_gpu_layers = -1;
  params.use_mmap = true;
  return params;
}

// ===== CONTEXT OPERATIONS =====

llama_context *llama_init_from_model(llama_model * /*model*/,
                                     llama_context_params /*params*/) {
  if (!g_stub_config.context_init_succeeds) {
    return nullptr;
  }

  // Generate unique fake pointer for this context instance
  uintptr_t fake_addr = g_context_counter.fetch_add(1);
  return reinterpret_cast<llama_context *>(fake_addr);
}

void llama_free(llama_context * /*ctx*/) {
  // Stub does nothing - no actual cleanup needed
}

uint32_t llama_n_ctx(const llama_context * /*ctx*/) {
  // Return default context size for stub tests
  return 512;
}

llama_context_params llama_context_default_params() {
  llama_context_params params;
  params.n_ctx = 512;
  params.n_batch = 512;
  return params;
}

// ===== KV CACHE MEMORY OPERATIONS =====

llama_memory_t llama_get_memory(llama_context * /*ctx*/) {
  return g_memory_handle;
}

bool llama_memory_seq_rm(llama_memory_t /*mem*/, llama_seq_id /*seq*/,
                         llama_pos /*p0*/, llama_pos /*p1*/) {
  return g_stub_config.rm_ok;
}

llama_pos llama_memory_seq_pos_max(llama_memory_t /*mem*/,
                                   llama_seq_id /*seq*/) {
  return g_stub_config.pos_max;
}

void llama_memory_clear(llama_memory_t /*mem*/, bool /*clear_kv*/) {
  // Stub: No-op for unit tests
  // Real behavior validated in integration tests
}

void llama_memory_seq_cp(llama_memory_t /*mem*/, llama_seq_id src,
                         llama_seq_id dst, llama_pos p0, llama_pos p1) {
  auto& cfg = llamaStubConfig();
  cfg.seq_cp_called = true;
  cfg.seq_cp_src = src;
  cfg.seq_cp_dst = dst;
  cfg.seq_cp_p0 = p0;
  cfg.seq_cp_p1 = p1;
}

void llama_memory_seq_keep(llama_memory_t /*mem*/, llama_seq_id seq) {
  auto& cfg = llamaStubConfig();
  cfg.seq_keep_called = true;
  cfg.seq_keep_seq = seq;
}

// ===== PER-SEQUENCE STATE OPERATIONS =====

size_t llama_state_seq_get_size(llama_context * /*ctx*/, llama_seq_id /*seq*/) {
  return g_stub_config.per_seq_size;
}

size_t llama_state_seq_get_data(llama_context * /*ctx*/, uint8_t * /*dst*/,
                                size_t /*size*/, llama_seq_id /*seq*/) {
  return g_stub_config.per_seq_rw;
}

size_t llama_state_seq_set_data(llama_context * /*ctx*/,
                                const uint8_t * /*src*/, size_t /*size*/,
                                llama_seq_id /*seq*/) {
  return g_stub_config.per_seq_rw;
}

// ===== FILE I/O OPERATIONS =====

// NOTE: Signature matches real llama.cpp (filepath before seq_id)
size_t llama_state_seq_save_file(llama_context * /*ctx*/,
                                  const char * /*filepath*/,
                                  llama_seq_id /*seq_id*/,
                                  const llama_token * /*tokens*/,
                                  size_t /*n_token_count*/) {
  if (!g_stub_config.file_operation_succeeds) {
    return 0;
  }
  return g_stub_config.file_write_bytes;
}

size_t llama_state_seq_load_file(llama_context * /*ctx*/,
                                  const char * /*filepath*/,
                                  llama_seq_id /*dest_seq_id*/,
                                  llama_token *tokens_out,
                                  size_t n_token_capacity,
                                  size_t *n_token_count_out) {
  if (!g_stub_config.file_operation_succeeds) {
    return 0;
  }

  // Simulate token restoration
  for (size_t i = 0; i < g_stub_config.file_token_count && i < n_token_capacity;
       ++i) {
    tokens_out[i] = 100 + static_cast<llama_token>(i); // Fake tokens: 100, 101, 102, ...
  }
  *n_token_count_out = g_stub_config.file_token_count;

  return g_stub_config.file_read_bytes;
}

// ===== GLOBAL STATE OPERATIONS (FALLBACK) =====

size_t llama_state_get_size(llama_context * /*ctx*/) {
  return g_stub_config.global_size;
}

size_t llama_state_get_data(llama_context * /*ctx*/, uint8_t * /*dst*/,
                            size_t /*size*/) {
  return g_stub_config.global_rw;
}

size_t llama_state_set_data(llama_context * /*ctx*/, const uint8_t * /*src*/,
                            size_t /*size*/) {
  return g_stub_config.global_rw;
}

// ===== BATCH OPERATIONS =====

llama_batch llama_batch_init(int32_t n_tokens, int32_t /*embd*/,
                             int32_t /*n_seq_max*/) {
  if (!g_stub_config.batch_init_succeeds) {
    // Return empty batch on failure
    return llama_batch{0, nullptr, nullptr, nullptr, nullptr, nullptr, 0};
  }

  // Allocate arrays for batch (simplified stub version)
  llama_batch batch;
  batch.n_tokens = 0;
  batch._capacity = n_tokens;

  // Allocate buffers
  batch.token = new llama_token[n_tokens];
  batch.pos = new int32_t[n_tokens];
  batch.n_seq_id = new int32_t[n_tokens];
  batch.seq_id = new llama_seq_id *[n_tokens];
  batch.logits = new int8_t[n_tokens];

  // Allocate seq_id arrays (max 1 sequence per token for simplicity)
  for (int32_t i = 0; i < n_tokens; ++i) {
    batch.seq_id[i] = new llama_seq_id[1];
  }

  return batch;
}

void llama_batch_free(llama_batch batch) {
  g_stub_config.batch_free_call_count++;

  if (batch.token) {
    // Free seq_id arrays first
    for (int32_t i = 0; i < batch._capacity; ++i) {
      delete[] batch.seq_id[i];
    }

    delete[] batch.token;
    delete[] batch.pos;
    delete[] batch.n_seq_id;
    delete[] batch.seq_id;
    delete[] batch.logits;
  }
}

// ===== DECODE OPERATIONS =====

int llama_decode(llama_context * /*ctx*/, llama_batch batch) {
  g_stub_config.decode_call_count++;

  // Track seq_id from batch (for multi-sequence tests)
  if (batch.n_tokens > 0 && batch.seq_id && batch.seq_id[0]) {
    llama_seq_id seq = batch.seq_id[0][0];  // First token's first seq_id
    g_stub_config.last_batch_seq_id = seq;

    // Track if all batches use consistent seq_id
    if (g_stub_config.all_batches_used_seq_id == -1) {
      // First batch - set the seq_id
      g_stub_config.all_batches_used_seq_id = seq;
    } else if (g_stub_config.all_batches_used_seq_id >= 0 &&
               g_stub_config.all_batches_used_seq_id != seq) {
      // Different seq_id than previous batches - mark as mixed
      g_stub_config.all_batches_used_seq_id = -2;
    }
    // If already -2 (mixed), leave it as -2
  }

  return g_stub_config.decode_result;
}

// ===== TOKENIZATION OPERATIONS =====

int llama_tokenize(const llama_vocab *vocab, const char *text, int32_t text_len,
                   llama_token *tokens, int32_t n_tokens_max,
                   bool /*add_special*/, bool /*parse_special*/
) {
  if (!vocab || !text || text_len <= 0) {
    return 0;
  }

  if (!g_stub_config.tokenize_succeeds) {
    return 0;
  }

  int32_t result_size =
      static_cast<int32_t>(g_stub_config.tokenize_result.size());

  // Two-pass algorithm: first call with tokens=nullptr returns negative size
  if (tokens == nullptr) {
    return -result_size;
  }

  // Second pass: copy tokens to buffer
  if (n_tokens_max < result_size) {
    return 0; // Buffer too small
  }

  for (int32_t i = 0; i < result_size; ++i) {
    tokens[i] = g_stub_config.tokenize_result[i];
  }

  return result_size;
}

int llama_detokenize(const llama_vocab *vocab, const llama_token *tokens,
                     int32_t n_tokens, char *text, int32_t text_len,
                     bool /*remove_special*/, bool /*unparse_special*/
) {
  if (!vocab || !tokens || n_tokens <= 0) {
    return 0;
  }

  if (!g_stub_config.detokenize_succeeds) {
    return 0;
  }

  int32_t result_size =
      static_cast<int32_t>(g_stub_config.detokenize_result.size());

  // Two-pass algorithm: first call with text=nullptr returns negative size
  if (text == nullptr) {
    return -result_size;
  }

  // Second pass: copy text to buffer
  if (text_len < result_size) {
    return -result_size; // Buffer too small, return negative to trigger resize
  }

  for (int32_t i = 0; i < result_size; ++i) {
    text[i] = g_stub_config.detokenize_result[i];
  }

  return result_size;
}

int llama_token_to_piece(const llama_vocab *vocab, llama_token /*token*/,
                         char *buf, int32_t length, int32_t /*lstrip*/,
                         bool /*special*/
) {
  if (!vocab) {
    return 0;
  }

  if (!g_stub_config.token_piece_succeeds) {
    return 0;
  }

  int32_t result_size = static_cast<int32_t>(g_stub_config.token_piece.size());

  // Two-pass algorithm: if buffer too small, return negative size
  if (length < result_size) {
    return -result_size;
  }

  // Copy piece to buffer
  for (int32_t i = 0; i < result_size; ++i) {
    buf[i] = g_stub_config.token_piece[i];
  }

  return result_size;
}

// ===== SAMPLING OPERATIONS =====

// Singleton vocab handle for get_vocab
static llama_vocab g_vocab_handle;

uint32_t llama_n_batch(const llama_context * /*ctx*/) {
  return 512;  // Default batch size
}

float *llama_get_logits_ith(llama_context * /*ctx*/, int32_t /*i*/) {
  if (g_stub_config.logits.empty()) {
    return nullptr;
  }
  return g_stub_config.logits.data();
}

int llama_vocab_n_tokens(const llama_vocab *vocab) {
  if (!vocab) {
    return 0;
  }
  return g_stub_config.vocab_size_value;
}

bool llama_vocab_is_eog(const llama_vocab *vocab, llama_token token) {
  if (!vocab) {
    return false;
  }
  return g_stub_config.eog_tokens.count(token) > 0;
}

const llama_vocab *llama_model_get_vocab(const llama_model *model) {
  if (!model) {
    return nullptr;
  }
  return &g_vocab_handle;
}

// ===== CHAT TEMPLATE AND SPECIAL TOKENS =====

const char *llama_model_chat_template(const llama_model *model,
                                      const char * /*name*/) {
  if (!model) {
    return nullptr;
  }
  if (g_stub_config.chat_template.empty()) {
    return nullptr;
  }
  return g_stub_config.chat_template.c_str();
}

llama_token llama_vocab_bos(const llama_vocab *vocab) {
  if (!vocab) {
    return LLAMA_TOKEN_NULL;
  }
  return g_stub_config.bos_token;
}

llama_token llama_vocab_eos(const llama_vocab *vocab) {
  if (!vocab) {
    return LLAMA_TOKEN_NULL;
  }
  return g_stub_config.eos_token;
}

llama_token llama_vocab_eot(const llama_vocab *vocab) {
  if (!vocab) {
    return LLAMA_TOKEN_NULL;
  }
  return g_stub_config.eot_token;
}

bool llama_vocab_get_add_bos(const llama_vocab *vocab) {
  if (!vocab) {
    return false;
  }
  // Return false by default (matches TinyLlama behavior for unit tests)
  return false;
}

bool llama_vocab_get_add_eos(const llama_vocab *vocab) {
  if (!vocab) {
    return false;
  }
  // Return false by default (matches TinyLlama behavior for unit tests)
  return false;
}

// ===== SAMPLER OPERATIONS (New API after commit b6451) =====

llama_sampler_chain_params llama_sampler_chain_default_params() {
  llama_sampler_chain_params params;
  params.no_perf = true;
  return params;
}

llama_sampler *llama_sampler_chain_init(llama_sampler_chain_params /*params*/) {
  return reinterpret_cast<llama_sampler *>(0x1000); // Stub pointer
}

void llama_sampler_chain_add(llama_sampler * /*chain*/,
                             llama_sampler * /*smpl*/) {
  // Stub - no-op
}

int llama_sampler_chain_n(llama_sampler * /*chain*/) {
  return 2; // Stub - return dummy chain length
}

llama_sampler *llama_sampler_chain_remove(llama_sampler * /*chain*/, int32_t /*i*/) {
  return reinterpret_cast<llama_sampler *>(0x5000); // Stub pointer
}

llama_sampler *llama_sampler_init_greedy() {
  return reinterpret_cast<llama_sampler *>(0x2000); // Stub pointer
}

llama_sampler *llama_sampler_init_temp(float /*temp*/) {
  return reinterpret_cast<llama_sampler *>(0x3000); // Stub pointer
}

llama_sampler *llama_sampler_init_dist(uint32_t /*seed*/) {
  return reinterpret_cast<llama_sampler *>(0x4000); // Stub pointer
}

llama_sampler *llama_sampler_init_top_k(int32_t /*k*/) {
  return reinterpret_cast<llama_sampler *>(0x5000); // Stub pointer
}

llama_sampler *llama_sampler_init_top_p(float /*p*/, size_t /*min_keep*/) {
  return reinterpret_cast<llama_sampler *>(0x6000); // Stub pointer
}

llama_sampler *llama_sampler_init_min_p(float /*p*/, size_t /*min_keep*/) {
  return reinterpret_cast<llama_sampler *>(0x7000); // Stub pointer
}

llama_sampler *llama_sampler_init_typical(float /*p*/, size_t /*min_keep*/) {
  return reinterpret_cast<llama_sampler *>(0x8000); // Stub pointer
}

llama_sampler *llama_sampler_init_penalties(int32_t /*penalty_last_n*/,
                                            float /*penalty_repeat*/,
                                            float /*penalty_freq*/,
                                            float /*penalty_present*/
) {
  return reinterpret_cast<llama_sampler *>(0x9000); // Stub pointer
}

llama_token llama_sampler_sample(llama_sampler * /*smpl*/,
                                 llama_context * /*ctx*/, int32_t /*idx*/) {
  return g_stub_config.sample_result; // Use existing stub config
}

void llama_sampler_free(llama_sampler * /*smpl*/) {
  // Stub - no-op
}

llama_sampler *llama_sampler_clone(llama_sampler *smpl) {
  auto& cfg = llamaStubConfig();
  cfg.sampler_clone_called = true;
  // Return null if input is null, otherwise return configured result
  return smpl ? cfg.sampler_clone_result : nullptr;
}

// ===== GRAMMAR SAMPLER OPERATIONS (For vendored common_sampler) =====

llama_sampler *llama_sampler_init_grammar(const llama_vocab * /*vocab*/,
                                          const char * /*grammar_str*/,
                                          const char * /*grammar_root*/) {
  return reinterpret_cast<llama_sampler *>(0xA000); // Stub pointer
}

void llama_sampler_apply(llama_sampler * /*smpl*/,
                         llama_token_data_array * /*cur_p*/) {
  // Stub - no-op
}

void llama_sampler_accept(llama_sampler * /*smpl*/, llama_token /*token*/) {
  // Stub - no-op
}

void llama_sampler_reset(llama_sampler * /*smpl*/) {
  // Stub - no-op
}

// ===== MODEL INTROSPECTION (For vendored common_sampler) =====

const llama_model *llama_get_model(const llama_context * /*ctx*/) {
  return reinterpret_cast<const llama_model *>(0xB000); // Stub pointer
}

// ===== EMBEDDING OPERATIONS =====

int32_t llama_model_n_embd(const llama_model * /*model*/) {
  return g_stub_config.n_embd;
}

int32_t llama_pooling_type(const llama_context * /*ctx*/) {
  return g_stub_config.pooling_type;
}

float *llama_get_embeddings(llama_context * /*ctx*/) {
  if (!g_stub_config.embeddings_available ||
      g_stub_config.embeddings.empty()) {
    return nullptr;
  }
  return g_stub_config.embeddings.data();
}

float *llama_get_embeddings_seq(llama_context * /*ctx*/, llama_seq_id /*seq*/) {
  if (!g_stub_config.embeddings_available ||
      g_stub_config.embeddings.empty()) {
    return nullptr;
  }
  return g_stub_config.embeddings.data();
}

float *llama_get_embeddings_ith(llama_context * /*ctx*/, int32_t /*idx*/) {
  if (!g_stub_config.embeddings_available ||
      g_stub_config.embeddings.empty()) {
    return nullptr;
  }
  return g_stub_config.embeddings.data();
}

} // extern "C"
