#pragma once

/**
 * liblloyal - Common definitions and logging
 *
 * Header-only library for llama.cpp-bound LLM operations
 * Version: 1.0.0 (bound to llama.cpp b6870)
 * License: MIT
 */

// ===== PLATFORM-NATIVE LOGGING =====

#ifdef DEBUG

#if defined(__APPLE__)
    #include <os/log.h>
    // Apple's Unified Logging System - integrates with Xcode Console
    #define LLOYAL_LOG_DEBUG(...) os_log(OS_LOG_DEFAULT, __VA_ARGS__)
    #define LLOYAL_LOG_INFO(...) os_log_info(OS_LOG_DEFAULT, __VA_ARGS__)
    #define LLOYAL_LOG_WARNING(...) os_log_error(OS_LOG_DEFAULT, __VA_ARGS__)
    #define LLOYAL_LOG_ERROR(...) os_log_fault(OS_LOG_DEFAULT, __VA_ARGS__)
#elif defined(__ANDROID__)
    #include <android/log.h>
    // Android's Logcat system - integrates with Android Studio
    #define LLOYAL_LOG_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, "lloyal", __VA_ARGS__)
    #define LLOYAL_LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, "lloyal", __VA_ARGS__)
    #define LLOYAL_LOG_WARNING(...) __android_log_print(ANDROID_LOG_WARN, "lloyal", __VA_ARGS__)
    #define LLOYAL_LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, "lloyal", __VA_ARGS__)
#else
    // Fallback for desktop/other platforms
    #include <cstdio>
    #define LLOYAL_LOG_DEBUG(...) printf(__VA_ARGS__)
    #define LLOYAL_LOG_INFO(...) printf(__VA_ARGS__)
    #define LLOYAL_LOG_WARNING(...) printf(__VA_ARGS__)
    #define LLOYAL_LOG_ERROR(...) printf(__VA_ARGS__)
#endif

#else
    // Release builds: compile out all logging (zero overhead)
    #define LLOYAL_LOG_DEBUG(...) do {} while (0)
    #define LLOYAL_LOG_INFO(...) do {} while (0)
    #define LLOYAL_LOG_WARNING(...) do {} while (0)
    #define LLOYAL_LOG_ERROR(...) do {} while (0)
#endif

// ===== CONSTANTS =====

namespace lloyal {

namespace defaults {
    // Context window size - reasonable for mobile devices
    static constexpr int N_CTX = 2048;

    // Batch size for context initialization - optimized for memory usage
    static constexpr int N_BATCH_INIT = 512;

    // Batch size for token processing - smaller batches for streaming
    static constexpr int N_BATCH_PROCESS = 32;
}

} // namespace lloyal
