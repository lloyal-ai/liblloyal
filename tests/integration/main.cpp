#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    // Parse --model flag to construct path from matrix directory
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            // Construct full path: ../../../models/matrix/<model-name>
            // (relative to build_integration directory)
            std::string model_name = argv[i + 1];
            std::string full_path = "../../../models/matrix/" + model_name;

            // Set environment variable for the test
            setenv("LLAMA_TEST_MODEL", full_path.c_str(), 1);

            // Remove --model and its argument from argv so doctest doesn't see them
            for (int j = i; j < argc - 2; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }

    // Run doctest with remaining arguments
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    int res = context.run();

    return res;
}
