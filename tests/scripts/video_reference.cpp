// Records how llama.cpp's own video helper tokenizes a clip: the reference a
// native video input must reproduce chunk for chunk. Built against a llama.cpp
// tree configured with MTMD_VIDEO=ON, which liblloyal's own builds never use.
//
//   video_reference <model.gguf> <mmproj.gguf> <clip.mp4> <prompt> > video-reference.json
//
// The prompt carries one media marker where the video goes.

#include <mtmd.h>
#include <mtmd-helper.h>
#include <llama.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static const char* chunk_type(mtmd_input_chunk_type type) {
  switch (type) {
    case MTMD_INPUT_CHUNK_TYPE_TEXT: return "text";
    case MTMD_INPUT_CHUNK_TYPE_IMAGE: return "image";
    case MTMD_INPUT_CHUNK_TYPE_AUDIO: return "audio";
  }
  return "unknown";
}

static std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
    else if (c == '\n') out += "\\n";
    else if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
    else out += static_cast<char>(c);
  }
  return out + "\"";
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <model> <mmproj> <clip> <prompt>\n", argv[0]);
    return 2;
  }
  llama_backend_init();
  auto mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0;
  llama_model* model = llama_model_load_from_file(argv[1], mparams);
  if (!model) return 1;

  auto cparams = mtmd_context_params_default();
  cparams.use_gpu = false;
  cparams.print_timings = false;
  cparams.warmup = false;
  mtmd_context* ctx = mtmd_init_from_file(argv[2], model, cparams);
  if (!ctx) return 1;

  std::ifstream file(argv[3], std::ios::binary);
  const std::vector<unsigned char> clip{std::istreambuf_iterator<char>(file), {}};
  auto wrap = mtmd_helper_bitmap_init_from_buf(ctx, clip.data(), clip.size(), false,
                                               mtmd_helper_init_opt_default());
  if (!wrap.bitmap) return 1;
  const auto info = mtmd_helper_video_get_info(wrap.video_ctx);

  const std::string prompt = argv[4];
  mtmd_input_text text{};
  text.text = prompt.c_str();
  text.text_len = prompt.size();
  text.add_special = false;
  text.parse_special = true;
  mtmd_input_chunks* chunks = mtmd_input_chunks_init();
  const mtmd_bitmap* bitmaps[] = {wrap.bitmap};
  if (mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) != 0) return 1;

  const auto* vocab = llama_model_get_vocab(model);
  std::printf("{\n  \"prompt\": %s,\n  \"fps\": %g,\n  \"width\": %u,\n  \"height\": %u,\n",
              json_string(prompt).c_str(), info.fps, info.width, info.height);
  std::printf("  \"n_tokens\": %zu,\n  \"n_pos\": %d,\n  \"chunks\": [\n",
              mtmd_helper_get_n_tokens(chunks), mtmd_helper_get_n_pos(chunks));
  const size_t n = mtmd_input_chunks_size(chunks);
  for (size_t i = 0; i < n; ++i) {
    const auto* chunk = mtmd_input_chunks_get(chunks, i);
    const auto type = mtmd_input_chunk_get_type(chunk);
    std::printf("    {\"type\": \"%s\", \"n_tokens\": %zu, \"n_pos\": %d", chunk_type(type),
                mtmd_input_chunk_get_n_tokens(chunk), mtmd_input_chunk_get_n_pos(chunk));
    if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n_text = 0;
      const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_text);
      std::string piece;
      std::printf(", \"tokens\": [");
      for (size_t k = 0; k < n_text; ++k) {
        char buf[256];
        const int len = llama_token_to_piece(vocab, tokens[k], buf, sizeof buf, 0, true);
        if (len > 0) piece.append(buf, static_cast<size_t>(len));
        std::printf("%s%d", k ? ", " : "", tokens[k]);
      }
      std::printf("], \"text\": %s", json_string(piece).c_str());
    }
    std::printf("}%s\n", i + 1 < n ? "," : "");
  }
  std::printf("  ]\n}\n");

  mtmd_input_chunks_free(chunks);
  mtmd_bitmap_free(wrap.bitmap);
  mtmd_helper_video_free(wrap.video_ctx);
  mtmd_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
