#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

struct llama_mtp_tree_node {
    llama_token token  = LLAMA_TOKEN_NULL;
    int32_t     parent = -1;
    int32_t     depth  = 0;
};

struct llama_mtp_tree_verify {
    // Packed verification rows, SGLang-style. Row 0 is normally the sampled
    // target token/root; following rows are candidate tree nodes.
    std::vector<llama_mtp_tree_node> nodes;

    // Optional retrieve metadata. These mirror SGLang's retrieve_index,
    // retrieve_next_token, and retrieve_next_sibling tables, but stay inert
    // until llama.cpp has a real tree-verify graph path.
    std::vector<int32_t> retrieve_index;
    std::vector<int32_t> retrieve_next_token;
    std::vector<int32_t> retrieve_next_sibling;

    // Filled by the target graph input path for packed verification batches.
    // Values are global KV indices (stream_offset + cell_index) for each row,
    // or -1 when a row is not present in that cache side.
    mutable std::vector<int64_t> kv_idxs_base;
    mutable std::vector<int64_t> kv_idxs_swa;

    int32_t n_steps = 0;
    int32_t top_k   = 0;

    bool empty() const {
        return nodes.empty();
    }
};

struct llama_mtp_tree_verify_result {
    std::vector<llama_token> accepted;
    llama_token  next       = LLAMA_TOKEN_NULL;
    int32_t      hidden_row = -1;
    int32_t      accepted_depth = 0;

    bool empty() const {
        return accepted.empty() && next == LLAMA_TOKEN_NULL;
    }
};

struct llama_mtp {
    llama_context * ctx_mtp    = nullptr; // non-owning
    std::vector<llama_context *> ctxs_mtp; // non-owning; one context per MTP layer
    llama_batch     hook_batch = {};

    // The last h-row of one ubatch needs the first token of the next ubatch to
    // form the next MTP pair, so keep it until the following contiguous batch.
    std::vector<float> pending_h;
    struct ggml_tensor * pending_t_gpu = nullptr; // D2D shortcut for generation
    llama_pos          pending_pos = -1;

    // Batch MTP prefill buffers
    std::vector<float>       prefill_h;
    std::vector<llama_token> prefill_tokens;
    std::vector<llama_pos>   prefill_pos;

    uint64_t hook_calls           = 0;
    uint64_t hook_skip_bad_shape  = 0;
    uint64_t hook_skip_caught_up  = 0;
    uint64_t hook_pending_breaks  = 0;
    uint64_t hook_gap_deferrals   = 0;
    uint64_t hook_decode_batches  = 0;
    uint64_t hook_decode_rows     = 0;
    uint64_t hook_decode_calls    = 0;
    uint64_t hook_decode_failures = 0;
};
