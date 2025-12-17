/**
 * lloyal/boundaries.hpp
 *
 * Boundary Detection for System 2 MCTS
 *
 * DESIGN: Two-phase draft/committed semantics
 * -----------------------------------------
 * From Stepper spec:
 * - feed_draft(token) called in produce() - SYNC
 * - reset_draft() called on implicit discard - SYNC
 * - commit_draft() called in commit() - SYNC
 * - clone() called in fork() - SYNC
 *
 * The BoundaryTracker maintains two parser states:
 * - Draft: advances speculatively during produce()
 * - Committed: only advances when commit() succeeds
 *
 * This prevents drift when tokens are discarded or resampled.
 *
 * DESIGN: Grammar-agnostic boundary mapping
 * ----------------------------------------
 * Remux emits structural events (node added, node completed).
 * BoundaryMapper (application-provided) interprets these as boundaries.
 *
 * Examples:
 * - Calibrate: "tool-start" → BoundaryInfo{kind: "tool", ...}
 * - System 2 MCTS: "paragraph completed" → BoundaryInfo{kind: "structural", ...}
 */

#pragma once

#include <remux/remux.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lloyal::boundaries {

// ============================================================================
// BoundaryInfo - Output from boundary detection
// ============================================================================

/**
 * BoundaryInfo - Describes a detected boundary
 *
 * From Stepper spec:
 *   type BoundaryInfo<K extends string = string> = {
 *     kind: K;
 *     pos: number;
 *     meta?: Record<string, unknown>;
 *   };
 *
 * Kind is application-defined: "word", "sentence", "structural", "tool", etc.
 */
struct BoundaryInfo {
    std::string kind;  // Application-defined boundary type
    size_t pos;        // Position in character stream
    std::unordered_map<std::string, std::string> meta;  // Optional payload
};

// ============================================================================
// BoundaryTracker Interface
// ============================================================================

/**
 * BoundaryTracker - Abstract interface matching Stepper spec
 *
 * From Stepper spec:
 *   type BoundaryTracker<K extends string = string> = {
 *     feedDraft(token: string): BoundaryInfo<K> | null;
 *     getDraftBoundary(): BoundaryInfo<K> | null;
 *     resetDraft(): void;
 *     commitDraft(): void;
 *     reset(): void;
 *   };
 *
 * Plus clone() for fork() support.
 *
 * All methods are SYNC - called within produce()/commit().
 */
class BoundaryTracker {
public:
    // ========================================================================
    // Draft Operations (called during produce() - SYNC)
    // ========================================================================

    /**
     * feed_draft - Process token, advance draft state
     *
     * Called in produce() with tokenToText() output.
     * Returns BoundaryInfo if a boundary was detected at this token.
     */
    virtual std::optional<BoundaryInfo> feed_draft(std::string_view token) = 0;

    /**
     * get_draft_boundary - Get current draft boundary (may not be committed)
     */
    virtual std::optional<BoundaryInfo> get_draft_boundary() const = 0;

    /**
     * reset_draft - Restore draft to committed state
     *
     * Called when produce() is invoked without prior commit() (implicit discard).
     */
    virtual void reset_draft() = 0;

    // ========================================================================
    // Commit Operations (called during commit() - SYNC)
    // ========================================================================

    /**
     * commit_draft - Promote draft state to committed
     *
     * Called in commit() after successful decode.
     */
    virtual void commit_draft() = 0;

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /**
     * reset - Full reset (both draft and committed)
     *
     * Called on stepper creation.
     */
    virtual void reset() = 0;

    // ========================================================================
    // Fork Support
    // ========================================================================

    /**
     * clone - Deep copy for fork()
     *
     * Must copy both draft and committed state.
     * O(1) when using structural sharing.
     */
    virtual std::unique_ptr<BoundaryTracker> clone() const = 0;

    virtual ~BoundaryTracker() = default;
};

// ============================================================================
// BoundaryMapper - Application-provided semantic interpretation
// ============================================================================

/**
 * BoundaryMapper - Function that interprets parser events as boundaries
 *
 * Remux provides structural events (ParserStepResult).
 * Mapper decides what constitutes a "boundary" for the application.
 *
 * Examples:
 * - Calibrate: Detect tool/think blocks, structural changes
 * - System 2: Detect semantic completion points for MCTS commits
 */
using BoundaryMapper = std::function<std::optional<BoundaryInfo>(
    const remux::block::ParserStepResult& result
)>;

// ============================================================================
// RemuxBoundaryTracker - BoundaryTracker using Remux parser
// ============================================================================

/**
 * RemuxBoundaryTracker - Boundary detection using Remux streaming parser
 *
 * Maintains two StreamingBlockParser instances:
 * - draft_parser_: Advances speculatively in feed_draft()
 * - committed_parser_: Only advances in commit_draft()
 *
 * fork() is O(1) due to immer-zipper structural sharing.
 */
class RemuxBoundaryTracker : public BoundaryTracker {
public:
    /**
     * Constructor
     *
     * @param grammar Remux grammar configuration
     * @param strategy Strategy step function for parser
     * @param mapper Application-provided boundary interpretation
     */
    RemuxBoundaryTracker(
        const remux::BlockGrammarConfig& grammar,
        remux::block::StrategyStepFn strategy,
        BoundaryMapper mapper
    )
        : grammar_(grammar)
        , strategy_(std::move(strategy))
        , mapper_(std::move(mapper))
        , draft_parser_(grammar_, strategy_)
        , committed_parser_(grammar_, strategy_)
    {}

    // ========================================================================
    // BoundaryTracker Interface Implementation
    // ========================================================================

    std::optional<BoundaryInfo> feed_draft(std::string_view token) override {
        // Feed each character to draft parser
        // (Lexer requires character-by-character for line/indent detection)
        for (char ch : token) {
            auto result = draft_parser_.step(ch, char_idx_++);
            if (result) {
                // Parser emitted event - ask mapper if it's a boundary
                if (auto boundary = mapper_(*result)) {
                    draft_boundary_ = boundary;
                }
            }
        }
        return draft_boundary_;
    }

    std::optional<BoundaryInfo> get_draft_boundary() const override {
        return draft_boundary_;
    }

    void reset_draft() override {
        // Restore draft to committed state - O(1) structural sharing
        draft_parser_ = committed_parser_.fork();
        draft_boundary_ = std::nullopt;
        char_idx_ = committed_char_idx_;
    }

    void commit_draft() override {
        // Promote draft to committed - O(1) structural sharing
        committed_parser_ = draft_parser_.fork();
        committed_char_idx_ = char_idx_;
        draft_boundary_ = std::nullopt;
    }

    void reset() override {
        draft_parser_ = remux::StreamingBlockParser(grammar_, strategy_);
        committed_parser_ = remux::StreamingBlockParser(grammar_, strategy_);
        draft_boundary_ = std::nullopt;
        char_idx_ = 0;
        committed_char_idx_ = 0;
    }

    std::unique_ptr<BoundaryTracker> clone() const override {
        auto cloned = std::make_unique<RemuxBoundaryTracker>(
            grammar_, strategy_, mapper_
        );
        cloned->draft_parser_ = draft_parser_.fork();
        cloned->committed_parser_ = committed_parser_.fork();
        cloned->draft_boundary_ = draft_boundary_;
        cloned->char_idx_ = char_idx_;
        cloned->committed_char_idx_ = committed_char_idx_;
        return cloned;
    }

    // ========================================================================
    // Structural Rollout (for verification oracles)
    // ========================================================================

    /**
     * structural_rollout - Complete open containers deterministically
     *
     * NOT LLM inference! Deterministic syntax completion via grammar rules:
     * - Reads zipper breadcrumbs to find unclosed containers
     * - Extracts closing delimiters from node metadata
     * - Appends closing text to complete syntax
     *
     * Cost: O(depth) string operations = microseconds (no LLM calls)
     * Use: Compilers/SAST tools need syntactically complete input
     *
     * Example:
     *   partial = "```python\nprint('hello')"
     *   zipper path = [root, fenced-code-block]
     *   node.metadata = {"fence": "```", "info": "python"}
     *   result = partial + "\n```\n" = "```python\nprint('hello')\n```\n"
     *
     * @param partial_text Text generated so far (may be incomplete)
     * @return Completed text with all containers properly closed
     */
    std::string structural_rollout(const std::string& partial_text) const {
        std::string result = partial_text;
        auto zipper = draft_parser_.state().zipper;

        // Collect closers from focus → root (inner → outer)
        std::vector<std::string> closers;

        while (zipper.focus().type != "root") {
            const auto& node = zipper.focus();

            // Check if container needs closing
            auto grouping_rule = find_grouping_rule(node.type);

            // Skip dedentation-closed containers (Python-style)
            if (grouping_rule && grouping_rule->dedentation_closes) {
                zipper = zipper.up();
                continue;
            }

            // Verbatim containers need explicit closers
            // Grammar-agnostic: Ask the grammar how to close this node
            if (grouping_rule && grouping_rule->is_verbatim_container) {
                // Try node first
                auto rule = find_block_rule(node.type);
                std::optional<std::string> closer;

                if (rule && rule->get_rollout) {
                    closer = rule->get_rollout(node);
                }

                // If node doesn't have rollout, try first child
                if (!closer && !node.children.empty()) {
                    const auto& first_child = node.children[0].get();
                    auto child_rule = find_block_rule(first_child.type);
                    if (child_rule && child_rule->get_rollout) {
                        closer = child_rule->get_rollout(first_child);
                    }
                }

                if (closer) {
                    closers.push_back(*closer);
                }
            }

            zipper = zipper.up();
        }

        // Append closers (inner → outer)
        for (const auto& closer : closers) {
            result += closer;
        }

        return result;
    }

    /**
     * get_draft_parser - Access draft parser zipper state
     *
     * Allows external code to inspect parser state for rollout or debugging.
     *
     * @return Current draft parser (contains zipper with breadcrumbs)
     */
    const remux::StreamingBlockParser& get_draft_parser() const {
        return draft_parser_;
    }

private:
    /**
     * find_grouping_rule - Query grammar for container type behavior
     *
     * Grammar-agnostic: Reads closing semantics from grammar config.
     * No hardcoded assumptions about what containers need closers.
     *
     * @param container_type Type to search for (e.g. "fenced-code-block")
     * @return Pointer to rule, or nullptr if not a container
     */
    const remux::BlockGroupingRule* find_grouping_rule(
        const std::string& container_type
    ) const {
        auto it = std::find_if(
            grammar_.block_grouping_rules.begin(),
            grammar_.block_grouping_rules.end(),
            [&container_type](const remux::BlockGroupingRule& rule) {
                return rule.container_type == container_type;
            }
        );
        return (it != grammar_.block_grouping_rules.end()) ? &(*it) : nullptr;
    }

    /**
     * find_block_rule - Query grammar for block rule by name
     *
     * Grammar-agnostic: Looks up rule definition for node type.
     *
     * @param rule_name Name of the rule (e.g. "fenced-code-start")
     * @return Pointer to rule, or nullptr if not found
     */
    const remux::BlockRule* find_block_rule(
        const std::string& rule_name
    ) const {
        for (const auto& [name, rule] : grammar_.block_rules) {
            if (name == rule_name) {
                return &rule;
            }
        }
        return nullptr;
    }

private:
    // Grammar and strategy (shared, not owned)
    remux::BlockGrammarConfig grammar_;
    remux::block::StrategyStepFn strategy_;

    // Boundary mapper (application-provided)
    BoundaryMapper mapper_;

    // Two parser states for draft/committed semantics
    remux::StreamingBlockParser draft_parser_;
    remux::StreamingBlockParser committed_parser_;

    // Current draft boundary (cleared on commit/reset)
    std::optional<BoundaryInfo> draft_boundary_;

    // Character position tracking
    size_t char_idx_ = 0;
    size_t committed_char_idx_ = 0;
};

// ============================================================================
// Boundary Mappers
// ============================================================================

/**
 * createStructuralBoundaryMapper - Grammar-agnostic boundary detection
 *
 * Detects structural boundaries from parser events, preserving ALL
 * grammar-specific metadata.
 *
 * Works with any grammar: CommonMark, Python, YAML, custom DSLs.
 *
 * Boundaries detected:
 * - ANY node added (block start)
 * - ANY node completed (block end, dedent)
 *
 * Metadata preserved:
 * - CommonMark: {type: "list-bullet", listMarker: "*", indent: "0"}
 * - Python: {type: "function-def", name: "foo", indent: "0"}
 * - YAML: {type: "mapping-item", key: "users", value: "...", indent: "2"}
 *
 * Applications can filter/interpret boundaries based on metadata.type.
 */
inline BoundaryMapper createStructuralBoundaryMapper() {
    return [](const remux::block::ParserStepResult& result)
        -> std::optional<BoundaryInfo>
    {
        // ANY node added is a structural boundary
        if (result.added) {
            const auto& node = result.added->get();

            // Copy ALL grammar-specific metadata
            auto meta = node.metadata;
            meta["type"] = node.type;
            meta["indent"] = std::to_string(node.indent);

            return BoundaryInfo{
                .kind = "structural",
                .pos = node.start_idx,
                .meta = meta
            };
        }

        // ANY node completed is a structural boundary
        if (!result.completed.empty()) {
            // Return first completed node
            const auto& node = result.completed[0].get();

            // Copy ALL grammar-specific metadata
            auto meta = node.metadata;
            meta["type"] = node.type + "-close";
            meta["indent"] = std::to_string(node.indent);

            return BoundaryInfo{
                .kind = "structural",
                .pos = node.end_idx,
                .meta = meta
            };
        }

        return std::nullopt;
    };
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * createCommonMarkBoundaryTracker - Create boundary tracker for CommonMark
 *
 * Uses explicit strategy with structural boundary mapper.
 * Detects boundaries at:
 * - Block starts (headings, lists, code blocks, etc.)
 * - Block ends (paragraph completion, list item close, etc.)
 *
 * Suitable for System 2 MCTS where action space = structural units.
 */
inline std::unique_ptr<BoundaryTracker> createCommonMarkBoundaryTracker() {
    auto grammar = remux::grammars::createCommonMarkGrammar();
    auto strategy = remux::strategies::createExplicitStrategy(grammar);
    auto mapper = createStructuralBoundaryMapper();

    return std::make_unique<RemuxBoundaryTracker>(
        grammar,
        std::move(strategy),
        std::move(mapper)
    );
}

/**
 * createCustomBoundaryTracker - Create boundary tracker with custom config
 *
 * @param grammar Remux grammar configuration
 * @param strategy Strategy step function for parser
 * @param mapper Application-provided boundary interpretation
 */
inline std::unique_ptr<BoundaryTracker> createCustomBoundaryTracker(
    const remux::BlockGrammarConfig& grammar,
    remux::block::StrategyStepFn strategy,
    BoundaryMapper mapper
) {
    return std::make_unique<RemuxBoundaryTracker>(
        grammar,
        std::move(strategy),
        std::move(mapper)
    );
}

} // namespace lloyal::boundaries
