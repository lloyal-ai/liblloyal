# liblloyal Usage Guide

Production patterns for building LLM applications with liblloyal.

---

## Table of Contents

- [Installation](#installation)
- [Basic Workflow](#basic-workflow)
- [Generation Patterns](#generation-patterns)
  - [Single-Turn Completion](#single-turn-completion)
  - [Multi-Turn Conversation](#multi-turn-conversation)
  - [Streaming Generation](#streaming-generation)
- [Cache Management Strategies](#cache-management-strategies)
  - [Sliding Window with Re-decode](#sliding-window-with-re-decode)
  - [State Checkpointing](#state-checkpointing)
  - [Multi-User Serving](#multi-user-serving)
- [Structured Output](#structured-output)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Installation

liblloyal is header-only. Integration steps:

1. **Add as submodule:**
```bash
git submodule add https://github.com/your-org/liblloyal.git vendor/liblloyal
git submodule update --init --recursive
```

2. **Configure build system:**

**CMake:**
```cmake
add_subdirectory(vendor/liblloyal/llama.cpp)
target_include_directories(your_target PRIVATE vendor/liblloyal/include)
target_link_libraries(your_target PRIVATE llama)
```

**Xcode:**
```ruby
s.header_mappings_dir = 'vendor/liblloyal/include'
s.vendored_frameworks = 'vendor/liblloyal/llama.cpp/ggml-metal.framework'
```

---

## Basic Workflow

Minimal example demonstrating complete inference pipeline:

```cpp
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/sampler.hpp>
#include <llama/llama.h>

int main() {
    llama_backend_init();

    // Load model (shared across contexts)
    auto model = lloyal::model_registry::acquire(
        "model.gguf",
        llama_model_default_params()
    );

    // Create inference context
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);

    // Tokenize input
    auto tokens = lloyal::tokenizer::tokenize(model.get(), "Explain transformers:");

    // Process through model
    lloyal::decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch);

    // Sample next token
    llama_token next = lloyal::sampler::greedy(ctx, model.get());

    // Convert to text
    std::string output = lloyal::tokenizer::detokenize(model.get(), next);

    llama_free(ctx);
    llama_backend_free();
}
```

---

## Generation Patterns

### Single-Turn Completion

Generate complete response to single prompt:

```cpp
std::string generate(
    llama_context* ctx,
    const llama_model* model,
    const std::string& prompt,
    int max_tokens = 100
) {
    lloyal::kv::clear_all(ctx);

    auto tokens = lloyal::tokenizer::tokenize(model, prompt);
    lloyal::decoder::decode_tokens(ctx, tokens, 0, 512);

    std::vector<llama_token> generated;
    int n_past = static_cast<int>(tokens.size());
    auto vocab = llama_model_get_vocab(model);

    for (int i = 0; i < max_tokens; ++i) {
        llama_token next = lloyal::sampler::greedy(ctx, model);

        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }

        generated.push_back(next);

        std::vector<llama_token> single = {next};
        lloyal::decoder::decode_tokens(ctx, single, n_past, 512);
        n_past++;
    }

    return lloyal::tokenizer::detokenize_batch(model, generated);
}
```

**Key points:**
- Clear KV cache between prompts
- Check EOS tokens to stop generation
- Increment `n_past` after each decode

### Multi-Turn Conversation

Maintain context across conversation turns:

```cpp
#include <lloyal/chat_template.hpp>

class ConversationSession {
private:
    llama_context* ctx_;
    const llama_model* model_;
    std::vector<Message> messages_;
    int n_past_ = 0;

public:
    std::string send(const std::string& user_input) {
        messages_.push_back({.role = "user", .content = user_input});

        // Format with chat template
        auto formatted = lloyal::chat_template::format(
            model_,
            nlohmann::json(messages_).dump(),
            ""  // Use model's built-in template
        );

        auto tokens = lloyal::tokenizer::tokenize(model_, formatted.prompt);

        // Decode only new tokens (optimization)
        std::vector<llama_token> new_tokens(
            tokens.begin() + n_past_,
            tokens.end()
        );

        if (!new_tokens.empty()) {
            lloyal::decoder::decode_tokens(ctx_, new_tokens, n_past_, 512);
            n_past_ = static_cast<int>(tokens.size());
        }

        // Generate response
        std::vector<llama_token> response_tokens;
        auto vocab = llama_model_get_vocab(model_);

        for (int i = 0; i < 200; ++i) {
            llama_token next = lloyal::sampler::greedy(ctx_, model_);

            if (llama_vocab_is_eog(vocab, next)) break;

            response_tokens.push_back(next);

            std::vector<llama_token> single = {next};
            lloyal::decoder::decode_tokens(ctx_, single, n_past_, 512);
            n_past_++;
        }

        std::string response = lloyal::tokenizer::detokenize_batch(
            model_, response_tokens
        );

        messages_.push_back({.role = "assistant", .content = response});
        return response;
    }
};
```

**Optimization:** Only decode new tokens on subsequent turns. KV cache retains prior conversation state.

### Streaming Generation

Progressive token output for responsive UX:

```cpp
void stream_generate(
    llama_context* ctx,
    const llama_model* model,
    const std::string& prompt,
    std::function<void(const std::string&)> on_token
) {
    auto tokens = lloyal::tokenizer::tokenize(model, prompt);
    lloyal::decoder::decode_tokens(ctx, tokens, 0, 512);

    int n_past = static_cast<int>(tokens.size());
    auto vocab = llama_model_get_vocab(model);

    for (int i = 0; i < 200; ++i) {
        llama_token next = lloyal::sampler::greedy(ctx, model);

        if (llama_vocab_is_eog(vocab, next)) break;

        // Immediate conversion for streaming
        std::string text = lloyal::tokenizer::detokenize(model, next);
        on_token(text);  // Fire callback (UI update, socket write, etc.)

        std::vector<llama_token> single = {next};
        lloyal::decoder::decode_tokens(ctx, single, n_past, 512);
        n_past++;
    }
}

// Usage
stream_generate(ctx, model, "Write a haiku:", [](const std::string& token) {
    std::cout << token << std::flush;
});
```

---

## Cache Management Strategies

### Sliding Window with Re-decode

**Problem:** Context window fills during long conversations.

**Solution:** StreamingLLM pattern—maintain fixed memory by keeping first N tokens (attention sinks) + last M tokens (recent context).

**Theory:** Xiao et al. (2023) demonstrated that transformer attention patterns develop "sinks" at initial positions. Maintaining these sinks + recent context preserves perplexity while enabling unbounded generation.

```cpp
class StreamingSession {
private:
    llama_context* ctx_;
    const llama_model* model_;
    int n_ctx_, n_batch_;

    std::vector<llama_token> conversation_;
    std::vector<llama_token> ORIGINAL_SINKS_;  // Immutable after capture
    int n_past_ = 0;

    const int SINK_COUNT = 4;
    const int TAIL_COUNT = 252;
    const int RESEED_THRESHOLD = 10;

public:
    void start(const std::string& initial_prompt) {
        conversation_ = lloyal::tokenizer::tokenize(model_, initial_prompt);

        if (conversation_.size() < SINK_COUNT) {
            throw std::runtime_error("Initial prompt must yield ≥4 tokens");
        }

        // CRITICAL: Capture sinks once, never modify
        ORIGINAL_SINKS_.assign(
            conversation_.begin(),
            conversation_.begin() + SINK_COUNT
        );

        lloyal::decoder::decode_tokens(ctx_, conversation_, 0, n_batch_);
        n_past_ = static_cast<int>(conversation_.size());
    }

    std::string generate_next(int max_tokens) {
        std::vector<llama_token> generated;
        auto vocab = llama_model_get_vocab(model_);

        for (int i = 0; i < max_tokens; ++i) {
            llama_pos current_pos = lloyal::kv::pos_max(ctx_, 0);

            if (current_pos >= n_ctx_ - RESEED_THRESHOLD) {
                reseed();
            }

            llama_token next = lloyal::sampler::greedy(ctx_, model_);

            if (llama_vocab_is_eog(vocab, next)) break;

            generated.push_back(next);
            conversation_.push_back(next);

            std::vector<llama_token> single = {next};
            lloyal::decoder::decode_tokens(ctx_, single, n_past_, n_batch_);
            n_past_++;
        }

        return lloyal::tokenizer::detokenize_batch(model_, generated);
    }

private:
    void reseed() {
        size_t tail_start = conversation_.size() > TAIL_COUNT
            ? conversation_.size() - TAIL_COUNT
            : 0;

        std::vector<llama_token> tail(
            conversation_.begin() + tail_start,
            conversation_.end()
        );

        // Use ORIGINAL_SINKS_ (not rolling window!)
        lloyal::kv::clear_and_reseed(ctx_, ORIGINAL_SINKS_, tail, n_batch_);

        n_past_ = SINK_COUNT + static_cast<int>(tail.size());
    }
};
```

**Critical invariant:** `ORIGINAL_SINKS_` must NEVER change. Using rolling "first 4" tokens violates positional bias learned during pretraining.

**Performance:** Empirical validation shows <10% perplexity increase (paper: 3.7%). See `tests/integration/clear_and_reseed_validation.cpp`.

### State Checkpointing

Serialize and restore KV cache for pause/resume or A/B testing:

```cpp
bool checkpoint_save(llama_context* ctx, const std::string& path) {
    size_t state_size = lloyal::kv::state_size(ctx, 0);
    if (state_size == 0) return false;

    std::vector<uint8_t> buffer(state_size);
    size_t saved = lloyal::kv::state_save(ctx, 0, buffer.data(), state_size);

    if (saved != state_size) return false;

    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(buffer.data()), state_size);

    return file.good();
}

bool checkpoint_load(llama_context* ctx, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    return lloyal::kv::state_load(ctx, 0, buffer.data(), size) > 0;
}
```

**Use cases:**
- Pause/resume conversations across app restarts
- Checkpoint expensive prefix computations
- A/B test different continuations from same state

### Multi-User Serving

Share model weights across independent user sessions:

```cpp
class InferenceService {
private:
    std::shared_ptr<llama_model> model_;
    std::unordered_map<std::string, llama_context*> contexts_;

public:
    InferenceService(const std::string& model_path) {
        model_ = lloyal::model_registry::acquire(
            model_path,
            llama_model_default_params()
        );
    }

    ~InferenceService() {
        for (auto& [user_id, ctx] : contexts_) {
            llama_free(ctx);
        }
    }

    bool create_session(const std::string& user_id) {
        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;

        llama_context* ctx = llama_init_from_model(model_.get(), ctx_params);
        if (!ctx) return false;

        contexts_[user_id] = ctx;
        return true;
    }

    std::string infer(const std::string& user_id, const std::string& prompt) {
        auto it = contexts_.find(user_id);
        if (it == contexts_.end()) return "";

        // Per-user isolated inference
        lloyal::kv::clear_all(it->second);
        auto tokens = lloyal::tokenizer::tokenize(model_.get(), prompt);
        lloyal::decoder::decode_tokens(it->second, tokens, 0, 512);

        llama_token next = lloyal::sampler::greedy(it->second, model_.get());
        return lloyal::tokenizer::detokenize(model_.get(), next);
    }
};
```

**Memory efficiency:** 1 model (~4GB) + N KV caches (~200MB each) instead of N full models.

---

### State Persistence (Save/Load Conversations)

**Problem:** Need to persist conversation state across app restarts, fork conversations at decision points, or share context between clients.

**Solution:** Use `kv::write_file()` and `kv::read_file()` to save/restore KV cache and tokens to disk.

**Use Cases:**
1. **Exit and Resume:** Save state before app termination, restore on next launch
2. **Conversation Forking:** Save at decision points, load to explore alternate paths
3. **Context Sharing:** Upload session file to cloud storage, share across devices/users

```cpp
#include <lloyal/kv.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>

class PersistentSession {
private:
    llama_context* ctx_;
    const llama_model* model_;
    std::vector<llama_token> conversation_;
    std::string session_file_;

public:
    // Start or resume conversation
    void init(const std::string& session_path) {
        session_file_ = session_path;

        // Try to restore existing session
        if (std::filesystem::exists(session_path)) {
            resume_from_file();
        } else {
            start_new_conversation();
        }
    }

    // Save current state before exit
    void save() {
        if (conversation_.empty()) return;

        size_t bytes = lloyal::kv::write_file(
            ctx_,
            0,
            session_file_,
            conversation_
        );

        if (bytes == 0) {
            throw std::runtime_error("Failed to save session");
        }

        std::cout << "Saved " << bytes << " bytes to " << session_file_ << std::endl;
    }

    // Resume conversation from saved file
    void resume_from_file() {
        auto data = lloyal::kv::read_file(ctx_, 0, session_file_);
        conversation_ = std::move(data.tokens);

        // Verify KV state restored correctly
        llama_pos max_pos = lloyal::kv::pos_max(ctx_, 0);
        assert(max_pos == static_cast<llama_pos>(conversation_.size() - 1));

        std::cout << "Resumed conversation with " << conversation_.size()
                  << " tokens" << std::endl;
    }

    // Generate response and append to conversation
    std::string generate_response(const std::string& user_input) {
        auto vocab = llama_model_get_vocab(model_);

        // Tokenize and decode user input
        auto input_tokens = lloyal::tokenizer::tokenize(vocab, user_input, false, false);
        lloyal::decoder::decode_tokens(ctx_, input_tokens, conversation_.size(), 512);
        conversation_.insert(conversation_.end(), input_tokens.begin(), input_tokens.end());

        // Generate response tokens
        std::vector<llama_token> response_tokens;
        for (int i = 0; i < 50; ++i) {
            llama_token next = lloyal::sampler::greedy(ctx_, vocab);

            if (lloyal::tokenizer::is_eog(vocab, next)) break;

            response_tokens.push_back(next);
            conversation_.push_back(next);

            // Decode single token
            std::vector<llama_token> single = {next};
            lloyal::decoder::decode_tokens(ctx_, single, conversation_.size() - 1, 512);
        }

        return lloyal::tokenizer::detokenize(vocab, response_tokens);
    }

private:
    void start_new_conversation() {
        lloyal::kv::clear_all(ctx_);
        conversation_.clear();
        std::cout << "Started new conversation" << std::endl;
    }
};
```

**Forking Example (Multiple Paths):**
```cpp
// Save state at decision point
std::vector<llama_token> conversation = {1, 100, 200, 300};
lloyal::decoder::decode_tokens(ctx, conversation, 0, 512);
lloyal::kv::write_file(ctx, 0, "fork_point.llama", conversation);

// Explore path A
auto response_a = generate("Path A prompt");
lloyal::kv::write_file(ctx, 0, "path_a.llama", conversation);

// Backtrack and explore path B
lloyal::kv::clear_all(ctx);
auto data = lloyal::kv::read_file(ctx, 0, "fork_point.llama");
auto response_b = generate("Path B prompt");
lloyal::kv::write_file(ctx, 0, "path_b.llama", conversation);
```

**Cloud Sharing Example (S3):**
```cpp
// Save locally
lloyal::kv::write_file(ctx, 0, "/tmp/session.llama", conversation);

// Upload to S3 (pseudo-code)
std::string signed_url = upload_to_s3("/tmp/session.llama");
// Share signed_url with other users/devices

// Other client downloads and loads
download_from_s3(signed_url, "/tmp/received_session.llama");
auto data = lloyal::kv::read_file(ctx, 0, "/tmp/received_session.llama");
// Continue generation with shared context
```

**Best Practices:**
- Save periodically during long conversations to prevent data loss
- Use descriptive filenames with timestamps: `session_2024-11-24_15-30.llama`
- Validate restored state by checking `kv::pos_max()` matches token count
- Session files are model-specific—ensure same model is used for save/load
- File size roughly equals KV cache size (context length × model dimensions)

---

## Structured Output

Force model to generate valid JSON using grammar constraints:

```cpp
#include <lloyal/grammar.hpp>

std::string generate_json(
    llama_context* ctx,
    const llama_model* model,
    const std::string& prompt,
    const nlohmann::json& schema
) {
    // Convert schema to GBNF grammar
    std::string grammar = lloyal::grammar::from_json_schema(schema.dump());

    // Create grammar sampler
    auto vocab = llama_model_get_vocab(model);
    llama_sampler* grammar_sampler = llama_sampler_init_grammar(
        vocab,
        grammar.c_str(),
        "root"
    );

    // Build sampler chain
    auto chain_params = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(chain, grammar_sampler);
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

    // Decode prompt
    auto tokens = lloyal::tokenizer::tokenize(model, prompt);
    lloyal::decoder::decode_tokens(ctx, tokens, 0, 512);
    int n_past = static_cast<int>(tokens.size());

    // Generate with grammar constraint
    std::vector<llama_token> generated;

    for (int i = 0; i < 500; ++i) {
        llama_token next = llama_sampler_sample(chain, ctx, -1);

        if (llama_vocab_is_eog(vocab, next)) break;

        generated.push_back(next);

        std::vector<llama_token> single = {next};
        lloyal::decoder::decode_tokens(ctx, single, n_past, 512);
        n_past++;
    }

    llama_sampler_free(chain);

    return lloyal::tokenizer::detokenize_batch(model, generated);
}

// Usage
nlohmann::json schema = {
    {"type", "object"},
    {"properties", {
        {"name", {{"type", "string"}}},
        {"age", {{"type", "integer"}}}
    }},
    {"required", {"name", "age"}}
};

std::string json_output = generate_json(ctx, model, "Generate user profile:", schema);
// Guaranteed valid JSON matching schema
```

**Guarantee:** Grammar sampling makes invalid syntax impossible (no malformed JSON).

---

## Best Practices

### Memory Management

**Efficient patterns:**
- Share models via `model_registry::acquire()` (ref-counted)
- Destroy idle contexts with `llama_free()`
- Use StreamingLLM for unbounded conversations

**Inefficient patterns:**
- Loading same model multiple times (wastes GB)
- Keeping unused contexts alive (leaks KV memory)
- Unbounded cache growth without reseeding

### Performance Tuning

**Key parameters:**

```cpp
// Mobile-optimized
ctx_params.n_ctx = 1024;
ctx_params.n_batch = 128;
ctx_params.n_threads = 2;
ctx_params.n_gpu_layers = -1;  // Use Metal/GPU

// Server-optimized
ctx_params.n_ctx = 4096;
ctx_params.n_batch = 512;
ctx_params.n_threads = 8;
ctx_params.n_gpu_layers = -1;  // Full GPU offload
```

**Parameter effects:**
- `n_ctx`: Larger = longer context, more KV memory
- `n_batch`: Larger = faster prompt processing, more decode memory
- `n_threads`: Match physical cores (diminishing returns beyond)
- `n_gpu_layers`: -1 for full offload (fastest), 0 for CPU-only

### Error Handling

Validate inputs and check return values:

```cpp
std::optional<std::string> safe_infer(
    llama_context* ctx,
    const llama_model* model,
    const std::string& prompt
) {
    try {
        if (!ctx || !model || prompt.empty()) {
            return std::nullopt;
        }

        auto tokens = lloyal::tokenizer::tokenize(model, prompt);
        if (tokens.empty()) return std::nullopt;

        lloyal::decoder::decode_tokens(ctx, tokens, 0, 512);
        llama_token next = lloyal::sampler::greedy(ctx, model);

        return lloyal::tokenizer::detokenize(model, next);

    } catch (const std::exception& e) {
        // Log error, return nullopt
        return std::nullopt;
    }
}
```

---

## Troubleshooting

### Model Load Failure

**Symptoms:** `model_registry::acquire()` returns `nullptr`

**Solutions:**
1. Verify file path is absolute and accessible
2. Confirm GGUF format (not GGML, PyTorch, etc.)
3. Check available disk space
4. Try default params (disable GPU offload initially)

```cpp
auto model = model_registry::acquire("model.gguf", llama_model_default_params());
if (!model) {
    // Debug: Try absolute path
    model = model_registry::acquire("/full/path/model.gguf", llama_model_default_params());
}
```

### Out of Memory

**Symptoms:** `llama_init_from_model()` returns `nullptr`

**Solutions:**
1. Reduce `n_ctx` (context window)
2. Reduce `n_batch` (batch size)
3. Use smaller quantized model (Q4_K_M vs F16)

```cpp
auto ctx_params = llama_context_default_params();
ctx_params.n_ctx = 512;   // Reduce from 2048
ctx_params.n_batch = 128;  // Reduce from 512

llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
```

### Perplexity Degradation After Reseed

**Symptoms:** Output quality drops after `clear_and_reseed()`

**Likely cause:** Changing sinks between reseeds

**Solution:** Verify sink immutability

```cpp
void reseed() {
    static bool first_call = true;
    static std::vector<llama_token> EXPECTED_SINKS;

    if (first_call) {
        EXPECTED_SINKS = ORIGINAL_SINKS_;
        first_call = false;
    } else {
        assert(ORIGINAL_SINKS_ == EXPECTED_SINKS);  // Verify invariant
    }

    auto tail = std::vector<llama_token>(conversation_.end() - 252, conversation_.end());
    lloyal::kv::clear_and_reseed(ctx_, ORIGINAL_SINKS_, tail, n_batch_);
}
```

### Decode Failure

**Symptoms:** `decode_tokens()` throws exception

**Causes:**
1. Null context or empty token vector
2. `n_past + tokens.size() > n_ctx`
3. Batch size exceeds context limit

**Debug:**
```cpp
try {
    decoder::decode_tokens(ctx, tokens, n_past, n_batch);
} catch (const std::exception& e) {
    std::cerr << "Decode error: " << e.what() << "\n";
    std::cerr << "  Context: " << (ctx ? "valid" : "null") << "\n";
    std::cerr << "  Tokens: " << tokens.size() << "\n";
    std::cerr << "  Position: " << n_past << "\n";
    std::cerr << "  Limit: " << n_ctx << "\n";

    if (n_past + tokens.size() > n_ctx) {
        std::cerr << "Position overflow—trigger reseed\n";
    }
}
```

---

## References

- StreamingLLM paper: Xiao et al. (2023) "Efficient Streaming Language Models with Attention Sinks"
- llama.cpp: https://github.com/ggerganov/llama.cpp
- API Reference: [api.md](./api.md)
- Integration tests: `tests/integration/`
