#define DOCTEST_CONFIG_IMPLEMENT
#include <cstdlib>
#include <doctest/doctest.h>
#include <llama/llama.h>

// Quiet log callback - suppresses all llama.cpp output
static void quiet_log_callback(enum ggml_log_level level, const char *text,
                               void *user_data) {
  (void)level;
  (void)text;
  (void)user_data;
  // Suppress all output
}

int main(int argc, char **argv) {
  // Suppress llama.cpp logging unless VERBOSE=1 is set
  const char *verbose = std::getenv("VERBOSE");
  if (!verbose || std::string(verbose) != "1") {
    llama_log_set(quiet_log_callback, nullptr);
  }

  doctest::Context context;
  context.applyCommandLine(argc, argv);
  return context.run();
}
