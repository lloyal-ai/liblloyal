// SPDX-License-Identifier: LicenseRef-FSL-1.1-Apache-2.0
// Copyright 2026 Lloyal Labs

// liblloyal engine-level benchmark, compiled unchanged against the pinned
// llama.cpp and against upstream master so the two builds report the same items
// in the same units. Only API that is identical at both revisions is used
// (branch/decode/mtmd/chat_in::format); chat_out is not benchmarked here.
//
// Usage: Bench <model.gguf> <mmproj.gguf> <image.jpg> [label]
// Output: one JSON object per line, {"label","item","unit","p50","p90","n",
// ...}
#include <lloyal/branch.hpp>
#include <lloyal/chat_in.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/mtmd.hpp>
#include <lloyal/tokenizer.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct Stats {
  double p50, p90, min, max;
  size_t n;
};
static Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const auto at = [&](double q) {
    return v[std::min(v.size() - 1, size_t(q * v.size()))];
  };
  return {at(0.5), at(0.9), v.front(), v.back(), v.size()};
}
static std::string g_label = "unlabelled";

// The label is user input printed inside a JSON string: escape what JSON
// escapes.
static std::string json_escaped(const std::string &s) {
  std::string out;
  for (unsigned char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
        out += buf;
      } else {
        out += static_cast<char>(c);
      }
    }
  }
  return out;
}
static void report(const char *item, const char *unit, const Stats &s,
                   const std::string &extra = "") {
  std::printf("{\"label\":\"%s\",\"item\":\"%s\",\"unit\":\"%s\",\"p50\":%.3f,"
              "\"p90\":%.3f,\"min\":%.3f,\"max\":%.3f,\"n\":%zu%s}\n",
              g_label.c_str(), item, unit, s.p50, s.p90, s.min, s.max, s.n,
              extra.c_str());
  std::fflush(stdout);
}
static void report_one(const char *item, const char *unit, double v,
                       const std::string &extra = "") {
  report(item, unit, Stats{v, v, v, v, 1}, extra);
}

struct Params {
  float temperature =
      0.0f; // greedy: deterministic token stream across revisions
  int32_t top_k = 40;
  float top_p = 0.95f;
  float typical_p = 1.0f;
  float min_p = 0.05f;
  float penalty_repeat = 1.0f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 42;
};

static std::vector<uint8_t> read_file(const char *path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static const char *REPORT_TOOL =
    R"([{"type":"function","function":{"name":"report","description":"Submit findings.","parameters":{"type":"object","properties":{"result":{"type":"string"},"sources":{"type":"array","items":{"type":"object","properties":{"title":{"type":"string"},"url":{"type":"string"}},"required":["title","url"]}}},"required":["result","sources"]}}}])";

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: Bench <model> <mmproj> <image> [label]\n");
    return 2;
  }
  if (argc > 4)
    g_label = json_escaped(argv[4]);
  llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
  llama_backend_init();

  // ---- load
  auto t0 = clk::now();
  auto mparams = llama_model_default_params();
  mparams.n_gpu_layers = 99;
  auto model = lloyal::ModelRegistry::acquire(argv[1], mparams);
  if (!model) {
    std::fprintf(stderr, "model load failed\n");
    return 1;
  }
  report_one("model_load", "ms", ms_since(t0));

  t0 = clk::now();
  auto cparams = llama_context_default_params();
  cparams.n_ctx = 32768;
  cparams.n_batch = 512;
  cparams.n_ubatch = 512;
  cparams.n_seq_max = 10;
  cparams.n_threads = 8;
  cparams.n_threads_batch = 8;
  llama_context *ctx = llama_init_from_model(model.get(), cparams);
  if (!ctx) {
    std::fprintf(stderr, "context init failed\n");
    return 1;
  }
  report_one("context_init", "ms", ms_since(t0));

  const llama_vocab *vocab = llama_model_get_vocab(model.get());
  const int n_vocab = llama_vocab_n_tokens(vocab);

  // ---- chat_in::format with a tool (the report grammar comes out of it)
  std::string grammar;
  {
    std::vector<double> xs;
    lloyal::chat_in::FormatResult fmt;
    for (int i = 0; i < 5; ++i) {
      lloyal::chat_in::FormatInputs in;
      in.messages_json =
          R"([{"role":"system","content":"You are a research agent."},{"role":"user","content":"Write a long report on the history of the bicycle."}])";
      in.tools_json = REPORT_TOOL;
      in.tool_choice = "required";
      in.enable_thinking = false;
      auto t = clk::now();
      fmt = lloyal::chat_in::format(model.get(), in);
      xs.push_back(ms_since(t));
    }
    grammar = fmt.grammar;
    report("chat_in_format_with_tool", "ms", stats(xs),
           ",\"grammar_bytes\":" + std::to_string(grammar.size()) +
               ",\"lazy\":" + (fmt.grammar_lazy ? "true" : "false"));
  }

  // ---- prefill
  lloyal::branch::BranchStore store;
  store.init_tenancy(ctx);
  std::string text;
  for (int i = 0; i < 40; ++i)
    text += "The safety bicycle displaced the penny-farthing in the "
            "eighteen-eighties, and cycling clubs spread across Europe. ";
  auto prompt = lloyal::tokenizer::tokenize(vocab, text, true, true);
  Params params;
  auto root = lloyal::branch::create(ctx, model.get(), store, 0, params, 512);
  t0 = clk::now();
  lloyal::branch::prefill(root, prompt.data(), prompt.size(), store);
  const double prefill_ms = ms_since(t0);
  report_one("prefill", "ms", prefill_ms,
             ",\"tokens\":" + std::to_string(prompt.size()) +
                 ",\"tok_per_s\":" +
                 std::to_string(prompt.size() / (prefill_ms / 1000.0)));

  // ---- fork cost
  {
    std::vector<double> xs;
    for (int i = 0; i < 8; ++i) {
      auto t = clk::now();
      auto h = lloyal::branch::fork(root, store);
      xs.push_back(ms_since(t));
      lloyal::branch::prune(h, store);
    }
    report("fork", "ms", stats(xs),
           ",\"prefix_tokens\":" + std::to_string(prompt.size()));
  }

  // ---- decode_each step for 1, 3, 8 branches (greedy, each branch feeds its
  // own token back)
  for (int n : {1, 3, 8}) {
    std::vector<lloyal::branch::BranchHandle> hs;
    for (int i = 0; i < n; ++i)
      hs.push_back(lloyal::branch::fork(root, store));
    std::vector<double> step_ms, sample_ms;
    for (int s = 0; s < 40; ++s) {
      std::vector<lloyal::branch::DecodeEachItem> items;
      // Only sample() is timed: acceptance and bookkeeping are not what this
      // row reports.
      double sampled_ms = 0.0;
      for (auto h : hs) {
        auto ts = clk::now();
        llama_token tok = lloyal::branch::sample(h, store);
        sampled_ms += ms_since(ts);
        lloyal::branch::accept_token(h, tok, store);
        items.push_back({h, tok});
      }
      sample_ms.push_back(sampled_ms / n);
      auto t = clk::now();
      store.decode_each(items);
      step_ms.push_back(ms_since(t));
    }
    const std::string extra = ",\"branches\":" + std::to_string(n);
    report((std::string("decode_each_step_n") + std::to_string(n)).c_str(),
           "ms", stats(step_ms), extra);
    report((std::string("sample_no_grammar_n") + std::to_string(n)).c_str(),
           "ms_per_branch", stats(sample_ms), extra);
    for (auto h : hs)
      lloyal::branch::prune(h, store);
  }

  // ---- sample() under the report grammar: mask-first and sample-first, one
  // branch, no decode in the loop
  for (bool sample_first : {false, true}) {
    auto h = lloyal::branch::fork(root, store);
    lloyal::branch::set_grammar(h, model.get(), grammar.c_str(), store);
    lloyal::branch::set_grammar_order(h, sample_first, store);
    std::vector<double> xs;
    for (int i = 0; i < 60; ++i) {
      auto t = clk::now();
      llama_token tok = lloyal::branch::sample(h, store);
      xs.push_back(ms_since(t));
      if (tok < 0)
        break;
      lloyal::branch::accept_token(h, tok, store);
      lloyal::branch::step(h, tok, store);
    }
    report(sample_first ? "sample_grammar_sample_first"
                        : "sample_grammar_mask_first",
           "ms", stats(xs), ",\"n_vocab\":" + std::to_string(n_vocab));
    lloyal::branch::prune(h, store);
  }

  // ---- image: tokenize+encode+decode one image through the segment kernel
  {
    auto mp = mtmd_context_params_default();
    mp.use_gpu = true;
    mp.print_timings = false;
    mp.n_threads = 8;
    mtmd_context *mctx = mtmd_init_from_file(argv[2], model.get(), mp);
    if (!mctx) {
      std::fprintf(stderr, "mmproj load failed\n");
      return 1;
    }
    auto bytes = read_file(argv[3]);
    const int32_t n_embd_inp = llama_model_n_embd_inp(model.get());
    std::vector<double> total_ms, construct_ms;
    int64_t cells = 0;
    llama_pos advance = 0;
    for (int i = 0; i < 3; ++i) {
      auto h = lloyal::branch::fork(root, store);
      std::string p =
          std::string("\n<|im_start|>user\n") + mtmd_default_marker() +
          "\nDescribe the picture.<|im_end|>\n<|im_start|>assistant\n";
      auto t = clk::now();
      lloyal::MtmdSource src(mctx, p, std::vector<std::vector<uint8_t>>{bytes},
                             std::span<const llama_token>{}, n_embd_inp);
      construct_ms.push_back(ms_since(t));
      auto r = store.decode_segments(h, src);
      total_ms.push_back(ms_since(t));
      cells = r.cells;
      advance = r.advance;
      lloyal::branch::prune(h, store);
    }
    report("image_prefill_total", "ms", stats(total_ms),
           ",\"cells\":" + std::to_string(cells) +
               ",\"advance\":" + std::to_string(advance));
    report("image_decode_and_tokenize", "ms", stats(construct_ms));
    mtmd_free(mctx);
  }

  lloyal::branch::prune(root, store);
  store.drain();
  llama_free(ctx);
  model.reset();
  llama_backend_free();
  return 0;
}
