#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>
#include <random>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_MTP,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"mtp",           COMMON_SPECULATIVE_TYPE_MTP},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    std::vector<float> draft_probs; // q(x) for each draft token, for p/q acceptance
    std::vector<std::vector<float>> draft_probs_all; // full draft softmax distribution per position, for rejection resampling
    std::vector<std::vector<float>> draft_logits; // raw draft logits per position, for logit blending

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void accept(uint16_t n_accepted) = 0;

    virtual common_speculative_preverified take_preverified() { return {}; }

    virtual common_speculative_tree_verify take_tree_verify() { return {}; }

    virtual common_speculative_preverified resolve_tree_verify(
            llama_context * ctx,
            const std::vector<int32_t> & row_idxs) {
        GGML_UNUSED(ctx);
        GGML_UNUSED(row_idxs);
        return {};
    }

    virtual int32_t n_max(const common_params_speculative & params) const = 0;
    virtual int32_t n_min(const common_params_speculative & params) const = 0;
};

struct common_speculative_checkpoint {
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;

    int64_t   n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    bool use_ckpt = false;
    common_speculative_checkpoint ckpt;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements,
            bool use_ckpt)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , use_ckpt(use_ckpt)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & /*prompt*/) override {
    }

    size_t create_checkpoint(int n_tokens_prompt) {
        int slot_id = 0;
        const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx_dft, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

        ckpt.pos_min  = llama_memory_seq_pos_min(llama_get_memory(ctx_dft), slot_id);
        ckpt.pos_max  = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot_id);
        ckpt.n_tokens = n_tokens_prompt;
        ckpt.data.resize(checkpoint_size);

        const size_t n = llama_state_seq_get_data_ext(ctx_dft, ckpt.data.data(), checkpoint_size, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (n != checkpoint_size) {
            GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
        }

        LOG_DBG("%s: pos_min = %d, pos_max = %d, size = %.3f MiB\n", __func__,
                ckpt.pos_min, ckpt.pos_max, (float) ckpt.data.size() / 1024 / 1024);
        return n;
    }

    size_t restore_checkpoint() {
        int slot_id = 0;
        LOG_DBG("%s: pos_min = %d, pos_max = %d\n", __func__, ckpt.pos_min, ckpt.pos_max);
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, ckpt.data.data(), ckpt.size(), slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (n != ckpt.size()) {
            GGML_ABORT("%s: failed to restore context checkpoint (pos_min=%d, pos_max=%d, size=%zu",
                        __func__, ckpt.pos_min, ckpt.pos_max, ckpt.size());
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, ckpt.pos_max + 1, -1);

        return n;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        const auto & sparams = params.draft;

        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0; // index of part to be reused in prompt_dft
        int reuse_n = 0; // length of part to be reused in prompt_dft

        const int n_ctx = llama_n_ctx(ctx_dft) - sparams.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        if (use_ckpt && i_start > 0) {
            LOG_WRN("%s: context shift is not supported with checkpoint-based contexts - skipping\n", __func__);
            return;
        }

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                   i       + cur < (int) prompt_dft.size() &&
                   prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }

            if (use_ckpt) {
                break;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, #prompt_dft = %zu, #prompt_cur = %zu\n",
                __func__, reuse_i, reuse_n, prompt_dft.size(), prompt_cur.size());
        if (use_ckpt && ckpt.n_tokens > reuse_n) {
            LOG_DBG("%s: checkpoint (n_tokens = %d) is outdated -> delete it\n", __func__, (int) ckpt.n_tokens);

            reuse_i = 0;
            reuse_n = 0;

            ckpt = {};
        }

        result.clear();
        result.reserve(sparams.n_max);

        if (reuse_n == 0 || (use_ckpt && reuse_i > 0)) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int64_t) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (sparams.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                GGML_ASSERT(!use_ckpt);

                bool is_removed = llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                if (!is_removed) {
                    LOG_ERR("%s: llama_memory_seq_rm failed, reuse_i=%d\n", __func__, reuse_i);
                    return;
                }
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                if (use_ckpt) {
                    if (ckpt.n_tokens > 0) {
                        LOG_DBG("%s: restoring checkpoint, reuse_n=%d, prompt_dft.size=%zu\n", __func__, reuse_n, prompt_dft.size());
                        restore_checkpoint();
                        reuse_n = ckpt.n_tokens;
                        prompt_dft.resize(reuse_n);
                    }
                } else {
                    const bool is_removed = llama_memory_seq_rm(mem_dft, 0, reuse_n, -1);
                    if (!is_removed) {
                        LOG_ERR("%s: llama_memory_seq_rm failed, reuse_n=%d, prompt_dft.size=%zu\n", __func__, reuse_n, prompt_dft.size());
                        return;
                    }
                    prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
                }
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());
            LOG_DBG("%s: draft prompt batch: %d tokens\n", __func__, batch.n_tokens);

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0 && ret != 1) {
                LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu\n",
                        __func__, ret, prompt_cur.size());
            }

            if (use_ckpt) {
                create_checkpoint(prompt_dft.size());
            }
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        //LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0 && ret != 1) {
            LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                    __func__, ret, prompt_cur.size(), prompt_dft.size());
        }

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < sparams.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < sparams.p_min) {
                break;
            }

            result.push_back(id);

            if (sparams.n_max <= (int) result.size()) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                        __func__, i, ret, prompt_cur.size(), prompt_dft.size());
            }

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t) sparams.n_max) {
                result.resize(sparams.n_max);
            }
        }

        if (result.size() < (size_t) sparams.n_min) {
            result.clear();
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }
};

struct common_speculative_state_mtp : public common_speculative_state {
    llama_context * ctx_tgt = nullptr;
    std::vector<llama_context *> ctxs_mtp;

    llama_batch      batch;
    common_sampler * smpl   = nullptr;
    int32_t          n_embd = 0;
    int32_t          n_vocab = 0;

    uint16_t last_n_drafted       = 0;
    int32_t  last_n_accepted      = -1;
    bool     chain_hidden         = false;
    bool     multi_ctx            = false;
    bool     multi_ctx_pos_local  = false;
    int32_t  draft_topk_log       = 0;
    int32_t  draft_topk_steps_rem = 0;
    int32_t  tree_probe_topk      = 0;
    int32_t  tree_probe_depth     = 2;
    int32_t  tree_probe_steps_rem = 0;
    int32_t  tree_probe_batch_cap = 64;
    int32_t  tree_probe_seq_cap   = 1;
    int32_t  tree_max_paths       = 0;
    bool     tree_probe_branch    = false;
    bool     tree_commit          = false;
    bool     tree_fast_commit     = false;
    bool     tree_trie_verify     = false;
    bool     tree_packed_verify   = false;
    bool     tree_scheduler_verify = false;
    bool     tree_timing          = false;
    bool     draft_timing         = false;
    bool     invariant_checks     = false;
    double   t_tree_hidden_ms     = 0.0;
    double   t_tree_mtp1_ms       = 0.0;
    double   t_tree_mtp2_ms       = 0.0;
    double   t_tree_topk_ms       = 0.0;
    double   t_tree_target_ms     = 0.0;
    double   t_tree_commit_ms     = 0.0;
    int32_t  n_tree_events        = 0;
    int32_t  n_tree_target_rows   = 0;
    int32_t  n_tree_mtp_rows      = 0;
    int32_t  n_tree_topk_calls    = 0;
    int32_t  forced_tgt_src_row    = -1;
    llama_batch tree_probe_batch;
    llama_batch branch_mtp_batch;

    std::vector<llama_pos> last_draft_pos;
    std::vector<llama_pos> tree_committed_draft_pos;
    common_speculative_preverified preverified;
    common_speculative_tree_verify pending_tree_verify;
    common_speculative_tree_verify active_tree_verify;

    common_speculative_state_mtp(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            std::vector<llama_context *> ctxs_mtp)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctxs_mtp(std::move(ctxs_mtp)) {
        GGML_ASSERT(ctx_tgt && !this->ctxs_mtp.empty());

        llama_context * ctx_mtp = this->ctxs_mtp.front();
        const llama_model * model_mtp = llama_get_model(ctx_mtp);
        n_embd = llama_model_n_embd(model_mtp);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_mtp));

        multi_ctx = this->ctxs_mtp.size() > 1;
        const char * chain_hidden_env = std::getenv("LLAMA_MIMO_MTP_CHAIN_HIDDEN");
        chain_hidden = false;
        if (chain_hidden_env != nullptr) {
            chain_hidden = std::strcmp(chain_hidden_env, "0") != 0;
        }

        const char * pos_mode_env = std::getenv("LLAMA_MIMO_MTP_MULTI_CTX_POS_MODE");
        multi_ctx_pos_local = multi_ctx;
        if (pos_mode_env != nullptr) {
            multi_ctx_pos_local = std::strcmp(pos_mode_env, "aligned") != 0;
        }

        if (const char * env = std::getenv("LLAMA_MIMO_MTP_DRAFT_TOPK_LOG")) {
            draft_topk_log = std::max(0, std::atoi(env));
        }
        draft_topk_steps_rem = draft_topk_log > 0 ? 8 : 0;
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_DRAFT_TOPK_STEPS")) {
            draft_topk_steps_rem = std::max(0, std::atoi(env));
        }

        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_PROBE_TOPK")) {
            tree_probe_topk = std::max(0, std::atoi(env));
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_DEPTH")) {
            tree_probe_depth = std::max(2, std::min(3, std::atoi(env)));
        }
        tree_probe_steps_rem = tree_probe_topk > 0 ? 4 : 0;
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_PROBE_STEPS")) {
            tree_probe_steps_rem = std::max(0, std::atoi(env));
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_MAX_PATHS")) {
            tree_max_paths = std::max(0, std::atoi(env));
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_BRANCH_PROBE")) {
            tree_probe_branch = std::strcmp(env, "0") != 0;
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_COMMIT")) {
            tree_commit = std::strcmp(env, "0") != 0;
            tree_probe_branch = tree_probe_branch || tree_commit;
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_FAST_COMMIT")) {
            tree_fast_commit = std::strcmp(env, "0") != 0;
            tree_commit = tree_commit || tree_fast_commit;
            tree_probe_branch = tree_probe_branch || tree_fast_commit;
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_TRIE")) {
            tree_trie_verify = std::strcmp(env, "0") != 0;
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_PACKED_VERIFY")) {
            tree_packed_verify = std::strcmp(env, "0") != 0;
            tree_trie_verify = tree_trie_verify || tree_packed_verify;
            tree_probe_branch = tree_probe_branch || tree_packed_verify;
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_SCHED_VERIFY")) {
            tree_scheduler_verify = std::strcmp(env, "0") != 0;
            tree_packed_verify = tree_packed_verify || tree_scheduler_verify;
            tree_trie_verify = tree_trie_verify || tree_scheduler_verify;
            tree_probe_branch = tree_probe_branch || tree_scheduler_verify;
            tree_commit = tree_commit || tree_scheduler_verify;
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_TREE_TIMING")) {
            tree_timing = std::strcmp(env, "0") != 0;
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_DRAFT_TIMING")) {
            draft_timing = std::strcmp(env, "0") != 0;
        }
        if (const char * env = std::getenv("LLAMA_MIMO_MTP_INVARIANTS")) {
            invariant_checks = std::strcmp(env, "0") != 0;
        }
        if (tree_probe_topk > 0) {
            int32_t n_paths = 1;
            for (int32_t i = 0; i < tree_probe_depth; ++i) {
                n_paths *= tree_probe_topk;
            }
            tree_probe_batch_cap = std::max(tree_probe_batch_cap, n_paths * (tree_probe_depth + 1));
            tree_probe_seq_cap   = std::max(tree_probe_seq_cap, n_paths);
        }

        LOG_INF("%s: MTP hidden chaining: %s; contexts: %zu; multi-ctx pos: %s\n",
                __func__, chain_hidden ? "on" : "off", this->ctxs_mtp.size(), multi_ctx_pos_local ? "local" : "aligned");

        {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 1;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            smpl = common_sampler_init(model_mtp, sparams);
        }

        batch = llama_batch_init(/*n_tokens=*/ 1, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        batch.token = (llama_token *) malloc(sizeof(llama_token));
        batch.n_tokens     = 1;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;

        tree_probe_batch = llama_batch_init(/*n_tokens=*/ tree_probe_batch_cap, /*embd=*/ 0, /*n_seq_max=*/ tree_probe_seq_cap);
        branch_mtp_batch = llama_batch_init(/*n_tokens=*/ tree_probe_batch_cap, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        branch_mtp_batch.token = (llama_token *) malloc(sizeof(llama_token) * tree_probe_batch_cap);

        // pre-allocate vocab-sized slabs for draft output (avoids per-step alloc)
        const int32_t slab_n = 4;
        draft_probs_all_slab.resize(slab_n);
        draft_logits_slab.resize(slab_n);
        for (auto & v : draft_probs_all_slab) v.resize(n_vocab);
        for (auto & v : draft_logits_slab) v.resize(n_vocab);

        llama_set_mtps(ctx_tgt, this->ctxs_mtp.data(), (int32_t) this->ctxs_mtp.size());
    }

    ~common_speculative_state_mtp() override {
        llama_set_mtps(ctx_tgt, nullptr, 0);
        llama_batch_free(branch_mtp_batch);
        llama_batch_free(tree_probe_batch);
        llama_batch_free(batch);
        common_sampler_free(smpl);
        for (llama_context * ctx_mtp : ctxs_mtp) {
            llama_free(ctx_mtp);
        }
    }

    void begin(const llama_tokens & prompt) override {
        last_n_accepted = -1;
        last_n_drafted  = 0;
        last_draft_pos.clear();
        forced_tgt_src_row = -1;
        active_tree_verify = {};
        pending_tree_verify = {};
        preverified = {};
        tree_committed_draft_pos.clear();
        // Reset tree timing counters so per-prompt logs are accurate
        t_tree_hidden_ms = 0.0;
        t_tree_mtp1_ms = 0.0;
        t_tree_mtp2_ms = 0.0;
        t_tree_topk_ms = 0.0;
        t_tree_target_ms = 0.0;
        t_tree_commit_ms = 0.0;
        n_tree_events = 0;
        n_tree_target_rows = 0;
        n_tree_mtp_rows = 0;
        n_tree_topk_calls = 0;

        const int32_t n = (int32_t) prompt.size();
        if (n <= 0) {
            return;
        }

        for (size_t i = 0; i < ctxs_mtp.size(); ++i) {
            const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctxs_mtp[i]), 0);
            if (pos_max < n - 1) {
                LOG_WRN("%s: ctx_mtp[%zu] pos_max=%d < N-1=%d; MTP prompt hook may be incomplete\n",
                        __func__, i, (int) pos_max, n - 1);
            }
        }
    }

    void rm_last_draft_tail(uint16_t first_to_drop) {
        for (uint16_t k = first_to_drop; k < last_n_drafted && k < last_draft_pos.size(); ++k) {
            llama_context * ctx_mtp = ctxs_mtp[multi_ctx ? std::min<size_t>(k, ctxs_mtp.size() - 1) : 0];
            const llama_pos pos = last_draft_pos[k];
            if (pos >= 0) {
                const bool ok = llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, pos, pos + 1);
                if (!ok) {
                    LOG_WRN("%s: failed to remove MTP draft tail at k=%d pos=%d; clearing ctx_mtp seq 0\n",
                            __func__, (int) k, (int) pos);
                    llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, -1, -1);
                } else if (invariant_checks) {
                    const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_mtp), 0);
                    if (pos_max >= pos) {
                        LOG_WRN("%s: MTP invariant: removed draft k=%d pos=%d but ctx_mtp pos_max=%d\n",
                                __func__, (int) k, (int) pos, (int) pos_max);
                    }
                }
            }
        }
    }

    std::vector<std::pair<float, llama_token>> collect_topk(llama_context * ctx_mtp, int32_t topk, int32_t row = 0, bool skip_sync = false) {
        if (!skip_sync) {
            llama_synchronize(ctx_mtp);
        }

        const llama_model * model = llama_get_model(ctx_mtp);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const float * logits = llama_get_logits_ith(ctx_mtp, row);
        if (logits == nullptr) {
            return {};
        }

        std::vector<std::pair<float, llama_token>> top;
        top.reserve((size_t) topk);

        for (llama_token tok = 0; tok < n_vocab; ++tok) {
            const float logit = logits[tok];
            if ((int32_t) top.size() < topk) {
                top.emplace_back(logit, tok);
                continue;
            }

            auto it_min = std::min_element(top.begin(), top.end(),
                    [](const auto & a, const auto & b) { return a.first < b.first; });
            if (it_min != top.end() && logit > it_min->first) {
                *it_min = { logit, tok };
            }
        }

        std::sort(top.begin(), top.end(),
                [](const auto & a, const auto & b) { return a.first > b.first; });

        return top;
    }

    llama_token fast_greedy_token(llama_context * ctx, int32_t row, bool skip_sync = false) {
        if (!skip_sync) {
            llama_synchronize(ctx);
        }
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const float * logits = llama_get_logits_ith(ctx, row);
        if (logits == nullptr) {
            return 0;
        }
        llama_token best = 0;
        float best_logit = logits[0];
        for (llama_token tok = 1; tok < n_vocab; ++tok) {
            if (logits[tok] > best_logit) {
                best_logit = logits[tok];
                best = tok;
            }
        }
        return best;
    }

    static float softmax_prob_token(const float * logits, int32_t n_vocab, llama_token token, float temperature) {
        float max_logit = -INFINITY;
        for (int32_t i = 0; i < n_vocab; i++) {
            max_logit = std::max(max_logit, logits[i] / temperature);
        }
        float sum = 0.0f;
        for (int32_t i = 0; i < n_vocab; i++) {
            sum += expf(logits[i] / temperature - max_logit);
        }
        return expf(logits[token] / temperature - max_logit) / sum;
    }

    std::vector<float> softmax_buf;
    std::vector<std::vector<float>> draft_probs_all_slab;
    std::vector<std::vector<float>> draft_logits_slab;

    llama_token sample_stochastic(llama_context * ctx, int32_t row, float temperature, float calib_temp, float & prob, bool store_logits = true) {
        static bool do_timing = std::getenv("LLAMA_SAMPLE_TIMING") != nullptr;
        const int64_t t0 = do_timing ? ggml_time_us() : 0;

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const float * logits = llama_get_logits_ith(ctx, row);
        if (logits == nullptr) {
            prob = 0.0f;
            return 0;
        }
        const int64_t t1 = do_timing ? ggml_time_us() : 0;

        if ((int32_t) softmax_buf.size() != n_vocab) {
            softmax_buf.resize(n_vocab);
        }
        float * buf = softmax_buf.data();

        const float inv_T = 1.0f / temperature;
#if defined(GGML_USE_ACCELERATE)
        vDSP_vsmul(logits, 1, &inv_T, buf, 1, n_vocab);
        float max_logit;
        vDSP_maxv(buf, 1, &max_logit, n_vocab);
        float neg_max = -max_logit;
        vDSP_vsadd(buf, 1, &neg_max, buf, 1, n_vocab);
        vvexpf(buf, buf, &n_vocab);
        float sum;
        vDSP_sve(buf, 1, &sum, n_vocab);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(buf, 1, &inv_sum, buf, 1, n_vocab);
#else
        float max_logit = -INFINITY;
        for (int32_t i = 0; i < n_vocab; i++) {
            buf[i] = logits[i] * inv_T;
            if (buf[i] > max_logit) max_logit = buf[i];
        }
        float sum = 0.0f;
        for (int32_t i = 0; i < n_vocab; i++) {
            buf[i] = expf(buf[i] - max_logit);
            sum += buf[i];
        }
        float inv_sum = 1.0f / sum;
        for (int32_t i = 0; i < n_vocab; i++) {
            buf[i] *= inv_sum;
        }
#endif

        const int64_t t2 = do_timing ? ggml_time_us() : 0;

        static thread_local std::mt19937 tl_rng{std::random_device{}()};
        float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(tl_rng);
        float cumulative = 0.0f;
        llama_token sampled = n_vocab - 1;
        for (int32_t i = 0; i < n_vocab; i++) {
            cumulative += buf[i];
            if (r < cumulative) {
                sampled = i;
                break;
            }
        }

        prob = (calib_temp == 1.0f) ? buf[sampled] : 0.0f;
        if (calib_temp != 1.0f) {
            const float calib_temperature = temperature * calib_temp;
            float max_logit_c = -INFINITY;
            for (int32_t i = 0; i < n_vocab; i++) {
                max_logit_c = std::max(max_logit_c, logits[i] / calib_temperature);
            }
            float sum_c = 0.0f;
            for (int32_t i = 0; i < n_vocab; i++) {
                sum_c += expf(logits[i] / calib_temperature - max_logit_c);
            }
            prob = expf(logits[sampled] / calib_temperature - max_logit_c) / sum_c;
        }

        {
            size_t idx = draft_probs_all.size();
            if (idx < draft_probs_all_slab.size()) {
                memcpy(draft_probs_all_slab[idx].data(), buf, n_vocab * sizeof(float));
                draft_probs_all.push_back(draft_probs_all_slab[idx]);
            } else {
                draft_probs_all.emplace_back(buf, buf + n_vocab);
            }
        }

        if (store_logits) {
            size_t idx = draft_logits.size();
            if (idx < draft_logits_slab.size()) {
                memcpy(draft_logits_slab[idx].data(), logits, n_vocab * sizeof(float));
                draft_logits.push_back(draft_logits_slab[idx]);
            } else {
                draft_logits.emplace_back(logits, logits + n_vocab);
            }
        }

        if (do_timing) {
            const int64_t t3 = ggml_time_us();
            static int count = 0;
            if (++count <= 10) {
                fprintf(stderr, "sample_timing: sync+logits=%.3f softmax=%.3f sample_copy=%.3f total=%.3f ms\n",
                    (t1-t0)/1000.0, (t2-t1)/1000.0, (t3-t2)/1000.0, (t3-t0)/1000.0);
            }
        }

        return sampled;
    }

    void log_draft_topk(int32_t k_step, const std::vector<std::pair<float, llama_token>> & top) {
        if (draft_topk_log <= 0 || draft_topk_steps_rem <= 0 || top.empty()) {
            return;
        }

        std::string msg;
        for (const auto & [logit, tok] : top) {
            if (!msg.empty()) {
                msg += ", ";
            }
            msg += std::to_string(tok);
            msg += ":";
            msg += std::to_string(logit);
        }

        LOG_INF("%s: draft top%d step=%d layer_ctx=%d candidates={%s}\n",
                __func__, draft_topk_log, k_step, multi_ctx ? k_step : 0, msg.c_str());

        draft_topk_steps_rem--;
    }

    llama_token greedy_token(llama_context * ctx, int32_t row) {
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const float * logits = llama_get_logits_ith(ctx, row);
        if (logits == nullptr) {
            return -1;
        }

        llama_token best = 0;
        float best_logit = logits[0];
        for (llama_token tok = 1; tok < n_vocab; ++tok) {
            if (logits[tok] > best_logit) {
                best_logit = logits[tok];
                best = tok;
            }
        }
        return best;
    }

    std::pair<float, llama_token> greedy_pair(llama_context * ctx, int32_t row) {
        const llama_token tok = greedy_token(ctx, row);
        const float * logits = llama_get_logits_ith(ctx, row);
        return { logits ? logits[tok] : 0.0f, tok };
    }

    struct tree_verify_result {
        std::vector<llama_token> path;
        int32_t accepted_depth = 0;
        int32_t branch_seq     = -1;
        llama_pos pos0         = -1;
        llama_token next_token = LLAMA_TOKEN_NULL;
        int32_t hidden_row     = -1;
        std::vector<int32_t> row_path;
    };

    struct scored_path {
        std::vector<llama_token> tokens;
        float score = 0.0f;
    };

    static bool path_has_prefix(const std::vector<llama_token> & path, const std::vector<llama_token> & prefix) {
        if (path.size() < prefix.size()) {
            return false;
        }
        for (size_t i = 0; i < prefix.size(); ++i) {
            if (path[i] != prefix[i]) {
                return false;
            }
        }
        return true;
    }

    std::vector<std::vector<llama_token>> prune_paths(std::vector<scored_path> scored) const {
        std::sort(scored.begin(), scored.end(),
                [](const scored_path & a, const scored_path & b) {
                    return a.score > b.score;
                });

        if (tree_max_paths > 0 && (int32_t) scored.size() > tree_max_paths) {
            scored.resize(tree_max_paths);
        }

        std::vector<std::vector<llama_token>> paths;
        paths.reserve(scored.size());
        for (auto & path : scored) {
            paths.push_back(std::move(path.tokens));
        }
        return paths;
    }

    struct trie_node {
        llama_token token = LLAMA_TOKEN_NULL;
        int32_t parent = -1;
        int32_t depth  = 0;
        int32_t row    = -1;
        std::vector<int32_t> children;
        std::vector<llama_seq_id> seq_ids;
    };

    int32_t trie_child_for(
            const std::vector<trie_node> & nodes,
            int32_t parent,
            llama_token token) const {
        for (const int32_t child : nodes[parent].children) {
            if (nodes[child].token == token) {
                return child;
            }
        }
        return -1;
    }

    std::vector<trie_node> build_trie_nodes(
            llama_token id_last,
            const std::vector<std::vector<llama_token>> & paths) const {
        std::vector<trie_node> nodes;
        nodes.push_back({});
        nodes[0].token = id_last;

        for (int32_t p = 0; p < (int32_t) paths.size(); ++p) {
            const llama_seq_id seq = p + 1;
            nodes[0].seq_ids.push_back(seq);

            int32_t cur = 0;
            for (int32_t k = 0; k < (int32_t) paths[p].size(); ++k) {
                int32_t child = trie_child_for(nodes, cur, paths[p][k]);
                if (child < 0) {
                    child = (int32_t) nodes.size();
                    nodes.push_back({});
                    nodes[child].token  = paths[p][k];
                    nodes[child].parent = cur;
                    nodes[child].depth  = nodes[cur].depth + 1;
                    nodes[cur].children.push_back(child);
                }
                nodes[child].seq_ids.push_back(seq);
                cur = child;
            }
        }

        return nodes;
    }

    bool prepare_scheduler_tree_verify(
            llama_token id_last,
            const std::vector<std::vector<llama_token>> & paths) {
        pending_tree_verify = {};

        if (paths.empty()) {
            return false;
        }

        const int32_t n_steps = (int32_t) paths[0].size();
        // tree_scheduler_verify env var is required for multi-step/tree scheduling,
        // but single-step linear verify (n_steps == 1) should work without it
        // because draft() already decided use_sched_verify for nmax <= 1.
        if (!tree_scheduler_verify && n_steps > 1) {
            return false;
        }
        const auto nodes = build_trie_nodes(id_last, paths);
        if (tree_probe_batch_cap < (int32_t) nodes.size()) {
            LOG_WRN("%s: scheduler tree batch too small for %zu rows\n", __func__, nodes.size());
            return false;
        }

        pending_tree_verify.tokens.reserve(nodes.size());
        pending_tree_verify.parents.reserve(nodes.size());
        pending_tree_verify.depths.reserve(nodes.size());
        for (const auto & node : nodes) {
            pending_tree_verify.tokens.push_back(node.token);
            pending_tree_verify.parents.push_back(node.parent);
            pending_tree_verify.depths.push_back(node.depth);
        }

        pending_tree_verify.paths   = paths;
        pending_tree_verify.n_steps = n_steps;
        pending_tree_verify.top_k   = tree_probe_topk;
        pending_tree_verify.in_place = tree_probe_topk <= 1;
        pending_tree_verify.packed   = tree_packed_verify;
        return true;
    }

    tree_verify_result tree_probe_trie_paths(llama_token id_last, const std::vector<std::vector<llama_token>> & paths, const char * label) {
        tree_verify_result best;
        const int32_t n_paths = (int32_t) paths.size();
        if (n_paths == 0) {
            return best;
        }
        const int32_t n_steps = (int32_t) paths[0].size();
        const uint32_t n_seq_max = llama_n_seq_max(ctx_tgt);
        const int32_t n_required_seq = tree_packed_verify ? 2 : n_paths + 1;
        if (n_seq_max < (uint32_t) n_required_seq) {
            LOG_WRN("%s: need n_seq_max >= %d for %d branch paths; set LLAMA_MIMO_MTP_TREE_SEQ_MAX=%d\n",
                    __func__, n_required_seq, n_paths, n_required_seq);
            tree_probe_steps_rem = 0;
            return best;
        }

        std::vector<trie_node> nodes = build_trie_nodes(id_last, paths);

        if (tree_probe_batch_cap < (int32_t) nodes.size()) {
            LOG_WRN("%s: tree trie batch too small for %zu rows\n", __func__, nodes.size());
            return best;
        }

        for (const auto & node : nodes) {
            if ((int32_t) node.seq_ids.size() > tree_probe_seq_cap) {
                LOG_WRN("%s: tree trie seq cap too small for %zu seq ids\n", __func__, node.seq_ids.size());
                return best;
            }
        }

        const llama_pos pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), 0) + 1;

        const int32_t n_probe_seqs = tree_packed_verify ? 1 : n_paths;
        for (int32_t s = 1; s <= n_probe_seqs; ++s) {
            llama_memory_seq_cp(llama_get_memory(ctx_tgt), 0, s, 0, -1);
        }

        common_batch_clear(tree_probe_batch);
        for (size_t i = 0; i < nodes.size(); ++i) {
            nodes[i].row = tree_probe_batch.n_tokens;
            common_batch_add(tree_probe_batch, nodes[i].token, pos0 + nodes[i].depth,
                    tree_packed_verify ? std::vector<llama_seq_id>{ 1 } : nodes[i].seq_ids, true);
        }

        std::vector<int32_t> parents;
        std::vector<int32_t> depths;
        if (tree_packed_verify) {
            parents.reserve(nodes.size());
            depths.reserve(nodes.size());
            for (const auto & node : nodes) {
                parents.push_back(node.parent);
                depths.push_back(node.depth);
            }
            if (!llama_set_mtp_tree_verify(ctx_tgt, parents.data(), depths.data(), (int32_t) parents.size(), n_steps, tree_probe_topk)) {
                LOG_WRN("%s: failed to enable packed MTP tree verify mask\n", __func__);
                llama_clear_mtp_tree_verify(ctx_tgt);
            }
        }

        const int64_t t_target_start = tree_timing ? ggml_time_us() : 0;
        const int32_t rc = llama_decode(ctx_tgt, tree_probe_batch);
        llama_clear_mtp_tree_verify(ctx_tgt);
        if (t_target_start != 0) {
            t_tree_target_ms += (ggml_time_us() - t_target_start) / 1000.0;
            n_tree_target_rows += tree_probe_batch.n_tokens;
        }

        if (rc != 0) {
            LOG_WRN("%s: llama_decode tree trie probe rc=%d\n", __func__, rc);
        } else {
            for (int32_t p = 0; p < n_paths; ++p) {
                std::string detail = "path=" + std::to_string(p);
                detail += " cand=[";
                for (size_t k = 0; k < paths[p].size(); ++k) {
                    if (k > 0) {
                        detail += ",";
                    }
                    detail += std::to_string(paths[p][k]);
                }
                detail += "] pred=[";

                int32_t cur = 0;
                int32_t accepted_depth = 0;
                for (int32_t k = 0; k < n_steps; ++k) {
                    const llama_token pred = greedy_token(ctx_tgt, nodes[cur].row);
                    if (k > 0) {
                        detail += ",";
                    }
                    detail += std::to_string(pred);
                    if (pred != paths[p][k]) {
                        break;
                    }

                    accepted_depth++;
                    cur = trie_child_for(nodes, cur, paths[p][k]);
                    if (cur < 0) {
                        break;
                    }
                }
                detail += "]";

                LOG_INF("%s: %s %s accepted_depth=%d/%d\n",
                        __func__, label, detail.c_str(), accepted_depth, n_steps);

                if (accepted_depth > best.accepted_depth && cur >= 0) {
                    best.accepted_depth = accepted_depth;
                    best.branch_seq     = p + 1;
                    best.pos0           = pos0;
                    best.next_token     = greedy_token(ctx_tgt, nodes[cur].row);
                    best.hidden_row     = nodes[cur].row;
                    best.path.assign(paths[p].begin(), paths[p].begin() + accepted_depth);
                    best.row_path.clear();
                    for (int32_t r = cur; r >= 0; r = nodes[r].parent) {
                        best.row_path.push_back(nodes[r].row);
                    }
                    std::reverse(best.row_path.begin(), best.row_path.end());
                }
            }
        }

        if (tree_packed_verify && tree_commit && best.accepted_depth > 0 && !best.row_path.empty()) {
            const int64_t t_commit_start = tree_timing ? ggml_time_us() : 0;
            if (llama_commit_mtp_tree_verify(ctx_tgt, best.row_path.data(), (int32_t) best.row_path.size())) {
                preverified.accepted = best.path;
                preverified.next     = best.next_token;
                forced_tgt_src_row   = best.hidden_row;
                LOG_INF("%s: packed-committed branch accepted_depth=%d next=%d rows=%zu\n",
                        __func__, best.accepted_depth, best.next_token, best.row_path.size());
            } else {
                LOG_WRN("%s: failed to packed-commit branch accepted_depth=%d\n",
                        __func__, best.accepted_depth);
            }
            if (t_commit_start != 0) {
                t_tree_commit_ms += (ggml_time_us() - t_commit_start) / 1000.0;
            }
        }

        if (!tree_packed_verify && tree_fast_commit && best.accepted_depth > 0 && best.branch_seq > 0) {
            auto * mem = llama_get_memory(ctx_tgt);
            const int64_t t_commit_start = tree_timing ? ggml_time_us() : 0;
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_memory_seq_cp(mem, best.branch_seq, 0, 0, -1);
            if (!llama_memory_seq_rm(mem, 0, best.pos0 + best.accepted_depth + 1, -1)) {
                LOG_WRN("%s: failed to trim fast-committed trie branch tail (pos=%d)\n",
                        __func__, (int) (best.pos0 + best.accepted_depth + 1));
            }
            if (t_commit_start != 0) {
                t_tree_commit_ms += (ggml_time_us() - t_commit_start) / 1000.0;
            }

            preverified.accepted = best.path;
            preverified.next     = best.next_token;
            forced_tgt_src_row   = best.hidden_row;
            LOG_INF("%s: fast-committed trie branch seq=%d accepted_depth=%d next=%d\n",
                    __func__, best.branch_seq, best.accepted_depth, best.next_token);
        }

        for (int32_t s = 1; s <= n_probe_seqs; ++s) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), s, -1, -1);
        }

        return best;
    }

    tree_verify_result tree_probe_paths(llama_token id_last, const std::vector<std::vector<llama_token>> & paths, const char * label) {
        if (tree_trie_verify) {
            return tree_probe_trie_paths(id_last, paths, label);
        }

        tree_verify_result best;
        const int32_t n_paths = (int32_t) paths.size();
        if (n_paths == 0) {
            return best;
        }
        const int32_t n_steps = (int32_t) paths[0].size();
        const uint32_t n_seq_max = llama_n_seq_max(ctx_tgt);
        if (n_seq_max <= (uint32_t) n_paths) {
            LOG_WRN("%s: need n_seq_max >= %d for %d branch paths; set LLAMA_MIMO_MTP_TREE_SEQ_MAX=%d\n",
                    __func__, n_paths + 1, n_paths, n_paths + 1);
            tree_probe_steps_rem = 0;
            return best;
        }

        if (tree_probe_batch_cap < n_paths * (n_steps + 1)) {
            LOG_WRN("%s: tree probe batch too small for %d rows\n", __func__, n_paths * (n_steps + 1));
            return best;
        }

        const llama_pos pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), 0) + 1;

        common_batch_clear(tree_probe_batch);
        int32_t seq = 1;
        for (const auto & path : paths) {
            llama_memory_seq_cp(llama_get_memory(ctx_tgt), 0, seq, 0, -1);
            common_batch_add(tree_probe_batch, id_last, pos0, { seq }, true);
            for (int32_t k = 0; k < n_steps; ++k) {
                common_batch_add(tree_probe_batch, path[k], pos0 + k + 1, { seq }, true);
            }
            seq++;
        }

        const int64_t t_target_start = tree_timing ? ggml_time_us() : 0;
        const int32_t rc = llama_decode(ctx_tgt, tree_probe_batch);
        if (t_target_start != 0) {
            t_tree_target_ms += (ggml_time_us() - t_target_start) / 1000.0;
            n_tree_target_rows += tree_probe_batch.n_tokens;
        }
        if (rc != 0) {
            LOG_WRN("%s: llama_decode tree probe rc=%d\n", __func__, rc);
        } else {
            for (int32_t p = 0; p < n_paths; ++p) {
                std::string detail = "path=" + std::to_string(p);
                detail += " cand=[";
                for (size_t k = 0; k < paths[p].size(); ++k) {
                    if (k > 0) {
                        detail += ",";
                    }
                    detail += std::to_string(paths[p][k]);
                }
                detail += "] pred=[";

                const int32_t row0 = p * (n_steps + 1);
                int32_t accepted_depth = 0;
                for (int32_t k = 0; k < n_steps; ++k) {
                    const llama_token pred = greedy_token(ctx_tgt, row0 + k);
                    if (k > 0) {
                        detail += ",";
                    }
                    detail += std::to_string(pred);
                    if (accepted_depth == k && pred == paths[p][k]) {
                        accepted_depth++;
                    }
                }
                detail += "]";

                LOG_INF("%s: %s %s accepted_depth=%d/%d\n",
                        __func__, label, detail.c_str(), accepted_depth, n_steps);

                if (accepted_depth > best.accepted_depth) {
                    best.accepted_depth = accepted_depth;
                    best.branch_seq     = p + 1;
                    best.pos0           = pos0;
                    best.next_token     = greedy_token(ctx_tgt, row0 + accepted_depth);
                    best.hidden_row     = row0 + accepted_depth;
                    best.path.assign(paths[p].begin(), paths[p].begin() + accepted_depth);
                }
            }
        }

        if (tree_fast_commit && best.accepted_depth > 0 && best.branch_seq > 0) {
            auto * mem = llama_get_memory(ctx_tgt);
            const int64_t t_commit_start = tree_timing ? ggml_time_us() : 0;
            // MiMo's iSWA KV cache only supports full cross-stream seq_cp. Copy the
            // winning branch whole, then trim any unaccepted branch tail on seq 0.
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_memory_seq_cp(mem, best.branch_seq, 0, 0, -1);
            if (!llama_memory_seq_rm(mem, 0, best.pos0 + best.accepted_depth + 1, -1)) {
                LOG_WRN("%s: failed to trim fast-committed branch tail (pos=%d)\n",
                        __func__, (int) (best.pos0 + best.accepted_depth + 1));
            }
            if (t_commit_start != 0) {
                t_tree_commit_ms += (ggml_time_us() - t_commit_start) / 1000.0;
            }

            preverified.accepted = best.path;
            preverified.next     = best.next_token;
            forced_tgt_src_row   = best.hidden_row;
            LOG_INF("%s: fast-committed branch seq=%d accepted_depth=%d next=%d\n",
                    __func__, best.branch_seq, best.accepted_depth, best.next_token);
        }

        for (int32_t s = 1; s <= n_paths; ++s) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), s, -1, -1);
        }

        return best;
    }

    void tree_probe(llama_token id_last, const std::vector<std::vector<std::pair<float, llama_token>>> & top_by_step) {
        if (tree_probe_topk <= 0 || tree_probe_steps_rem <= 0 || top_by_step.empty() || top_by_step[0].empty()) {
            return;
        }

        const int32_t n_steps = std::min<int32_t>(2, (int32_t) top_by_step.size());
        const int32_t n0 = std::min<int32_t>(tree_probe_topk, (int32_t) top_by_step[0].size());
        const int32_t n1 = n_steps > 1 ? std::min<int32_t>(tree_probe_topk, (int32_t) top_by_step[1].size()) : 1;

        std::vector<std::vector<llama_token>> paths;
        paths.reserve((size_t) n0 * n1);
        for (int32_t i = 0; i < n0; ++i) {
            for (int32_t j = 0; j < n1; ++j) {
                std::vector<llama_token> path;
                path.push_back(top_by_step[0][i].second);
                if (n_steps > 1) {
                    path.push_back(top_by_step[1][j].second);
                }
                paths.push_back(std::move(path));
            }
        }

        tree_probe_paths(id_last, paths, "linear");

        tree_probe_steps_rem--;
    }

    tree_verify_result branch_conditioned_tree_probe(
            llama_token id_last,
            ggml_tensor * tgt_src,
            int32_t src_row,
            const std::vector<std::pair<float, llama_token>> & top0) {
        tree_verify_result best;
        if (!tree_probe_branch || !multi_ctx || ctxs_mtp.size() < 2 || tgt_src == nullptr || top0.empty()) {
            return best;
        }
        if (tree_probe_topk <= 0 || tree_probe_steps_rem <= 0) {
            return best;
        }

        const int32_t n0 = std::min<int32_t>(tree_probe_topk, (int32_t) top0.size());
        const uint32_t n_seq_max = llama_n_seq_max(ctxs_mtp[1]);
        if (n_seq_max <= (uint32_t) n0) {
            LOG_WRN("%s: need MTP n_seq_max >= %d for %d parent branches; set LLAMA_MIMO_MTP_TREE_SEQ_MAX=%d\n",
                    __func__, n0 + 1, n0, n0 + 1);
            return best;
        }
        if (tree_probe_batch_cap < n0) {
            LOG_WRN("%s: branch MTP batch too small for %d rows\n", __func__, n0);
            return best;
        }

        llama_context * ctx_mtp = ctxs_mtp[1];
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::vector<float> h((size_t) n_embd);

        const int64_t t_hidden_start = tree_timing ? ggml_time_us() : 0;
        llama_synchronize(ctx_tgt);
        ggml_backend_tensor_get(tgt_src, h.data(), (size_t) src_row * row_bytes, row_bytes);
        if (t_hidden_start != 0) {
            t_tree_hidden_ms += (ggml_time_us() - t_hidden_start) / 1000.0;
        }

        common_batch_clear(branch_mtp_batch);
        const llama_pos pos1 = llama_memory_seq_pos_max(llama_get_memory(ctx_mtp), 0) + 1;
        for (int32_t i = 0; i < n0; ++i) {
            const int32_t seq = i + 1;
            llama_memory_seq_cp(llama_get_memory(ctx_mtp), 0, seq, 0, -1);
            std::memcpy(branch_mtp_batch.embd + (size_t) i * n_embd, h.data(), row_bytes);
            branch_mtp_batch.token[i]     = top0[i].second;
            branch_mtp_batch.pos[i]       = pos1;
            branch_mtp_batch.n_seq_id[i]  = 1;
            branch_mtp_batch.seq_id[i][0] = seq;
            branch_mtp_batch.logits[i]    = 1;
        }
        branch_mtp_batch.n_tokens = n0;

        llama_set_mtp_layer_idx(ctx_mtp, 1);
        const int64_t t_mtp1_start = tree_timing ? ggml_time_us() : 0;
        const int32_t rc = llama_decode(ctx_mtp, branch_mtp_batch);
        if (t_mtp1_start != 0) {
            t_tree_mtp1_ms += (ggml_time_us() - t_mtp1_start) / 1000.0;
            n_tree_mtp_rows += branch_mtp_batch.n_tokens;
        }
        if (rc != 0) {
            LOG_WRN("%s: llama_decode branch MTP rc=%d\n", __func__, rc);
        } else {
            std::vector<scored_path> scored_paths;
            std::vector<scored_path> partial_paths;
            scored_paths.reserve((size_t) n0 * tree_probe_topk);
            partial_paths.reserve((size_t) n0 * tree_probe_topk);
            {
                // One sync for all top-k collections in this batch
                llama_synchronize(ctx_mtp);
                for (int32_t i = 0; i < n0; ++i) {
                    const int64_t t_topk_start = tree_timing ? ggml_time_us() : 0;
                    const auto top1 = tree_probe_topk == 1
                        ? std::vector<std::pair<float, llama_token>>{ greedy_pair(ctx_mtp, i) }
                        : collect_topk(ctx_mtp, tree_probe_topk, i, true);
                    if (t_topk_start != 0) {
                    t_tree_topk_ms += (ggml_time_us() - t_topk_start) / 1000.0;
                    n_tree_topk_calls++;
                }

                std::string msg;
                for (const auto & [logit, tok] : top1) {
                    if (!msg.empty()) {
                        msg += ", ";
                    }
                    msg += std::to_string(tok);
                    msg += ":";
                    msg += std::to_string(logit);
                }
                LOG_INF("%s: parent=%d token=%d layer_ctx=1 candidates={%s}\n",
                        __func__, i, top0[i].second, msg.c_str());

                for (const auto & cand : top1) {
                    partial_paths.push_back({ { top0[i].second, cand.second }, top0[i].first + cand.first });
                }
            }
            }

            if (tree_probe_depth >= 3 && ctxs_mtp.size() >= 3 && !partial_paths.empty()) {
                llama_context * ctx_mtp2 = ctxs_mtp[2];
                const int32_t n_pairs = (int32_t) partial_paths.size();
                const uint32_t n_seq_max2 = llama_n_seq_max(ctx_mtp2);
                if (n_seq_max2 <= (uint32_t) n_pairs) {
                    LOG_WRN("%s: need MTP n_seq_max >= %d for %d depth-2 branches; set LLAMA_MIMO_MTP_TREE_SEQ_MAX=%d\n",
                            __func__, n_pairs + 1, n_pairs, n_pairs + 1);
                    scored_paths = std::move(partial_paths);
                } else if (tree_probe_batch_cap < n_pairs) {
                    LOG_WRN("%s: branch MTP batch too small for %d depth-2 rows\n", __func__, n_pairs);
                    scored_paths = std::move(partial_paths);
                } else {
                    common_batch_clear(branch_mtp_batch);
                    const llama_pos pos2 = llama_memory_seq_pos_max(llama_get_memory(ctx_mtp2), 0) + 1;
                    for (int32_t i = 0; i < n_pairs; ++i) {
                        const int32_t seq = i + 1;
                        llama_memory_seq_cp(llama_get_memory(ctx_mtp2), 0, seq, 0, -1);
                        std::memcpy(branch_mtp_batch.embd + (size_t) i * n_embd, h.data(), row_bytes);
                        branch_mtp_batch.token[i]     = partial_paths[i].tokens[1];
                        branch_mtp_batch.pos[i]       = pos2;
                        branch_mtp_batch.n_seq_id[i]  = 1;
                        branch_mtp_batch.seq_id[i][0] = seq;
                        branch_mtp_batch.logits[i]    = 1;
                    }
                    branch_mtp_batch.n_tokens = n_pairs;

                    llama_set_mtp_layer_idx(ctx_mtp2, 2);
                    const int64_t t_mtp2_start = tree_timing ? ggml_time_us() : 0;
                    const int32_t rc2 = llama_decode(ctx_mtp2, branch_mtp_batch);
                    if (t_mtp2_start != 0) {
                        t_tree_mtp2_ms += (ggml_time_us() - t_mtp2_start) / 1000.0;
                        n_tree_mtp_rows += branch_mtp_batch.n_tokens;
                    }
                    if (rc2 != 0) {
                        LOG_WRN("%s: llama_decode branch MTP depth-3 rc=%d\n", __func__, rc2);
                        scored_paths = std::move(partial_paths);
                    } else {
                        scored_paths.reserve((size_t) n_pairs * tree_probe_topk);
                        llama_synchronize(ctx_mtp2);
                        for (int32_t i = 0; i < n_pairs; ++i) {
                            const int64_t t_topk_start = tree_timing ? ggml_time_us() : 0;
                            const auto top2 = tree_probe_topk == 1
                                ? std::vector<std::pair<float, llama_token>>{ greedy_pair(ctx_mtp2, i) }
                                : collect_topk(ctx_mtp2, tree_probe_topk, i, true);
                            if (t_topk_start != 0) {
                                t_tree_topk_ms += (ggml_time_us() - t_topk_start) / 1000.0;
                                n_tree_topk_calls++;
                            }

                            std::string msg;
                            for (const auto & [logit, tok] : top2) {
                                if (!msg.empty()) {
                                    msg += ", ";
                                }
                                msg += std::to_string(tok);
                                msg += ":";
                                msg += std::to_string(logit);
                            }
                            LOG_INF("%s: depth2_parent=%d tokens=[%d,%d] layer_ctx=2 candidates={%s}\n",
                                    __func__, i, partial_paths[i].tokens[0], partial_paths[i].tokens[1], msg.c_str());

                            for (const auto & cand : top2) {
                                scored_paths.push_back({
                                    { partial_paths[i].tokens[0], partial_paths[i].tokens[1], cand.second },
                                    partial_paths[i].score + cand.first
                                });
                            }
                        }
                    }
                }
            } else {
                scored_paths = std::move(partial_paths);
            }

            const auto paths = prune_paths(std::move(scored_paths));
            if (tree_scheduler_verify && prepare_scheduler_tree_verify(id_last, paths)) {
                best.accepted_depth = -1;
                if (tree_timing) {
                    n_tree_events++;
                    LOG_CNT("%s: prepared scheduler tree verify event=%d rows(target=%zu, mtp=%d, topk_calls=%d)\n",
                            __func__, n_tree_events, pending_tree_verify.tokens.size(), n_tree_mtp_rows, n_tree_topk_calls);
                }
            } else
            best = tree_probe_paths(id_last, paths, tree_probe_depth >= 3 ? "branch3" : "branch");
            const bool should_commit_mtp_branch_state = tree_fast_commit || (tree_packed_verify && tree_commit);
            if (should_commit_mtp_branch_state && best.accepted_depth > 0) {
                tree_committed_draft_pos.clear();
                if (!last_draft_pos.empty()) {
                    tree_committed_draft_pos.push_back(last_draft_pos[0]);
                }

                if (best.accepted_depth >= 2) {
                    int32_t parent_seq = -1;
                    for (int32_t i = 0; i < n0; ++i) {
                        if (top0[i].second == best.path[0]) {
                            parent_seq = i + 1;
                            break;
                        }
                    }
                    if (parent_seq > 0) {
                        llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, -1, -1);
                        llama_memory_seq_cp(llama_get_memory(ctx_mtp), parent_seq, 0, 0, -1);
                        tree_committed_draft_pos.push_back(llama_memory_seq_pos_max(llama_get_memory(ctx_mtp), 0));
                    } else {
                        LOG_WRN("%s: could not find MTP layer-1 branch seq for accepted token %d\n",
                                __func__, best.path[0]);
                    }
                }

                if (best.accepted_depth >= 3 && ctxs_mtp.size() >= 3) {
                    llama_context * ctx_mtp2 = ctxs_mtp[2];
                    int32_t pair_seq = -1;
                    for (int32_t i = 0; i < (int32_t) partial_paths.size(); ++i) {
                        if (path_has_prefix(partial_paths[i].tokens, { best.path[0], best.path[1] })) {
                            pair_seq = i + 1;
                            break;
                        }
                    }
                    if (pair_seq > 0) {
                        llama_memory_seq_rm(llama_get_memory(ctx_mtp2), 0, -1, -1);
                        llama_memory_seq_cp(llama_get_memory(ctx_mtp2), pair_seq, 0, 0, -1);
                        tree_committed_draft_pos.push_back(llama_memory_seq_pos_max(llama_get_memory(ctx_mtp2), 0));
                    } else {
                        LOG_WRN("%s: could not find MTP layer-2 branch seq for accepted prefix [%d,%d]\n",
                                __func__, best.path[0], best.path[1]);
                    }
                }
            }
            if (tree_timing) {
                n_tree_events++;
                LOG_CNT("%s: tree timing events=%d rows(target=%d, mtp=%d, topk_calls=%d) hidden %.2f ms, mtp1 %.2f ms, mtp2 %.2f ms, topk %.2f ms, target %.2f ms, commit %.2f ms\n",
                        __func__, n_tree_events, n_tree_target_rows, n_tree_mtp_rows, n_tree_topk_calls,
                        t_tree_hidden_ms, t_tree_mtp1_ms, t_tree_mtp2_ms,
                        t_tree_topk_ms, t_tree_target_ms, t_tree_commit_ms);
            }
        }

        for (int32_t seq = 1; seq <= n0; ++seq) {
            llama_memory_seq_rm(llama_get_memory(ctx_mtp), seq, -1, -1);
        }

        return best;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        GGML_UNUSED(prompt_tgt);
        draft_tokens.clear();

        llama_mtp_flush(ctx_tgt);

        if (last_n_drafted > 0) {
            rm_last_draft_tail(0);
            last_n_drafted  = 0;
            last_n_accepted = -1;
            last_draft_pos.clear();
        }

        int32_t n_max = multi_ctx
            ? std::max(1, std::min<int32_t>(params.draft.n_max, (int32_t) ctxs_mtp.size()))
            : std::max(1, params.draft.n_max);

        const size_t  row_bytes = (size_t) n_embd * sizeof(float);

        const bool use_sched_verify = n_max <= 1 || tree_scheduler_verify;

        llama_token cond_tok = id_last;
        llama_pos   pos_base = llama_memory_seq_pos_max(llama_get_memory(ctxs_mtp.front()), 0) + 1;

        ggml_tensor * tgt_src     = nullptr;
        int32_t       tgt_src_row = 0;
        bool          used_forced_tgt_src_row = false;
        std::vector<std::vector<std::pair<float, llama_token>>> top_by_step;

        int64_t t_draft_hidden_us = 0;
        int64_t t_draft_decode_us = 0;
        int64_t t_draft_sample_us = 0;

        for (int32_t k = 0; k < n_max; ++k) {
            llama_context * ctx_mtp = ctxs_mtp[multi_ctx ? (size_t) k : 0];
            ggml_tensor * src;
            int32_t       src_row;
            if (k == 0 || !chain_hidden) {
                src = llama_context_get_t_h_pre_norm(ctx_tgt);
                if (last_n_accepted < 0) {
                    src_row = (src && src->ne[1] > 0) ? (int32_t) src->ne[1] - 1 : 0;
                } else if (forced_tgt_src_row >= 0) {
                    src_row = forced_tgt_src_row;
                    used_forced_tgt_src_row = true;
                } else {
                    src_row = last_n_accepted;
                }
                if (k == 0) {
                    tgt_src     = src;
                    tgt_src_row = src_row;
                } else if (tgt_src != nullptr) {
                    src     = tgt_src;
                    src_row = tgt_src_row;
                }
            } else {
                llama_context * src_ctx = multi_ctx ? ctxs_mtp[k - 1] : ctx_mtp;
                src = llama_context_get_t_mtp_out(src_ctx);
                src_row = src ? (int32_t) src->ne[1] - 1 : 0;
            }

            if (!src) {
                LOG_WRN("%s: missing MTP source tensor at k=%d; stopping draft\n", __func__, k);
                return;
            }

            {
                const int64_t t_step_start = ggml_time_us();
                if (k == 0 || !chain_hidden) {
                    // For multi-ctx non-chained mode, all steps read the same target
                    // hidden state. Only sync on the first step; reuse CPU buffer.
                    if (k == 0) {
                        llama_synchronize(ctx_tgt);
                    }
                    ggml_backend_tensor_get(src, batch.embd, (size_t) src_row * row_bytes, row_bytes);
                } else {
                    llama_context * src_ctx = multi_ctx ? ctxs_mtp[k - 1] : ctx_mtp;
                    llama_synchronize(src_ctx);
                    ggml_backend_tensor_get(src, batch.embd, (size_t) src_row * row_bytes, row_bytes);
                }
                t_draft_hidden_us += ggml_time_us() - t_step_start;
            }

            batch.token[0] = cond_tok;
            batch.pos[0]   = (multi_ctx && multi_ctx_pos_local)
                ? llama_memory_seq_pos_max(llama_get_memory(ctx_mtp), 0) + 1
                : pos_base + k;

            llama_set_mtp_layer_idx(ctx_mtp, k);
            const llama_pos pos_before = invariant_checks ? llama_memory_seq_pos_max(llama_get_memory(ctx_mtp), 0) : -1;

            {
                const int64_t t_step_start = ggml_time_us();
                const int32_t dec_rc = llama_decode(ctx_mtp, batch);
                t_draft_decode_us += ggml_time_us() - t_step_start;
                if (dec_rc != 0) {
                    LOG_DBG("%s: llama_decode(ctx_mtp) rc=%d at k=%d; stopping draft\n", __func__, dec_rc, k);
                    return;
                }
            }
            if (invariant_checks) {
                const llama_pos pos_after = llama_memory_seq_pos_max(llama_get_memory(ctx_mtp), 0);
                if (pos_after < batch.pos[0]) {
                    LOG_WRN("%s: MTP invariant: draft k=%d pos=%d did not advance ctx_mtp pos_max (%d -> %d)\n",
                            __func__, k, (int) batch.pos[0], (int) pos_before, (int) pos_after);
                }
                if (pos_before >= batch.pos[0]) {
                    LOG_WRN("%s: MTP invariant: draft k=%d overwrote/nonmonotonic pos=%d with prior pos_max=%d\n",
                            __func__, k, (int) batch.pos[0], (int) pos_before);
                }
            }

            const int32_t n_tree_collect = (use_sched_verify && tree_probe_topk <= 1) ? 0 : tree_probe_topk;
            const int32_t n_top_collect = std::max(draft_topk_log, n_tree_collect);
            // Fast draft sampling: collect_topk() syncs; if we need top-k logging
            // we use its synced state and do a cheap greedy argmax without
            // running the full CPU sampler chain (saves ~3 ms/step).
            const bool use_fast_draft = true; // env-gated if needed later
            const float temperature = params.temperature;
            const float calib_temp  = params.calib_temp;
            std::vector<std::pair<float, llama_token>> top;
            llama_token best = LLAMA_TOKEN_NULL;
            if (n_top_collect > 0) {
                top = collect_topk(ctx_mtp, n_top_collect, 0, false);
                if (!top.empty()) {
                    best = top[0].second;
                }
            }
            if (n_tree_collect > 0) {
                top_by_step.push_back(top);
            }
            log_draft_topk(k, top);

            {
                const int64_t t_step_start = ggml_time_us();
                float q_x = 1.0f; // draft probability for sampled token
                if (temperature > 0.0f) {
                    best = sample_stochastic(ctx_mtp, 0, temperature, calib_temp, q_x, params.logit_blend > 0.0f);
                    common_sampler_accept(smpl, best, /*accept_grammar=*/ false);
                } else if (best == LLAMA_TOKEN_NULL) {
                    best = fast_greedy_token(ctx_mtp, 0, false);
                } else {
                    common_sampler_accept(smpl, best, /*accept_grammar=*/ false);
                }
                t_draft_sample_us += ggml_time_us() - t_step_start;
                draft_tokens.push_back(best);
                draft_probs.push_back(q_x);
                last_draft_pos.push_back(batch.pos[0]);
                cond_tok = best;
            }
        }

        if (draft_timing && n_max > 0) {
            LOG_INF("%s: draft timing n=%d hidden %.3f ms, decode %.3f ms, sample %.3f ms\n",
                    __func__, n_max,
                    t_draft_hidden_us / 1000.0,
                    t_draft_decode_us / 1000.0,
                    t_draft_sample_us / 1000.0);
        }

        tree_verify_result branch_best;
        bool ran_packed_tree_verify = false;
        if (use_sched_verify && !draft_tokens.empty()) {
            std::vector<std::vector<llama_token>> paths;
            if (tree_probe_topk > 1 && !top_by_step.empty()) {
                // EAGLE-style depth-1 multi-branch tree drafting
                for (const auto & tok_pair : top_by_step[0]) {
                    paths.push_back({tok_pair.second});
                }
            } else {
                paths.push_back(draft_tokens);
            }
            if (prepare_scheduler_tree_verify(id_last, paths)) {
                branch_best.accepted_depth = -1;
                ran_packed_tree_verify = true;
            }
        } else if (!top_by_step.empty()) {
            branch_best = branch_conditioned_tree_probe(id_last, tgt_src, tgt_src_row, top_by_step[0]);
            ran_packed_tree_verify = tree_packed_verify && tree_probe_branch && branch_best.accepted_depth >= 0;
        }
        if (tree_commit || ran_packed_tree_verify) {
            if (tree_probe_steps_rem > 0) {
                tree_probe_steps_rem--;
            }
        } else {
            tree_probe(id_last, top_by_step);
        }

        const bool branch_commit_succeeded = tree_commit && branch_best.accepted_depth > 0 && (!tree_packed_verify || !preverified.empty());
        if (branch_commit_succeeded) {
            rm_last_draft_tail((uint16_t) branch_best.accepted_depth);
            draft_tokens = branch_best.path;
            last_draft_pos.clear();
            if (tree_committed_draft_pos.size() >= draft_tokens.size()) {
                last_draft_pos.assign(tree_committed_draft_pos.begin(), tree_committed_draft_pos.begin() + draft_tokens.size());
            } else {
                const llama_pos pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), 0) + 2;
                for (size_t i = 0; i < draft_tokens.size(); ++i) {
                    last_draft_pos.push_back(pos0 + (llama_pos) i);
                }
            }
            tree_committed_draft_pos.clear();
            if (tree_timing) {
                LOG_INF("%s: tree commit overriding linear draft with accepted_depth=%d tokens=[%s]\n",
                        __func__, branch_best.accepted_depth, string_from(ctx_tgt, draft_tokens).c_str());
            }
        }

        last_n_drafted = (uint16_t) draft_tokens.size();
        if (used_forced_tgt_src_row) {
            forced_tgt_src_row = -1;
        }
    }

    common_speculative_preverified take_preverified() override {
        common_speculative_preverified result = std::move(preverified);
        preverified = {};
        return result;
    }

    common_speculative_tree_verify take_tree_verify() override {
        if (!draft_probs.empty()) {
            pending_tree_verify = {};
            return {};
        }
        active_tree_verify = pending_tree_verify;
        common_speculative_tree_verify result = std::move(pending_tree_verify);
        pending_tree_verify = {};
        return result;
    }

    common_speculative_preverified resolve_tree_verify(
            llama_context * ctx,
            const std::vector<int32_t> & row_idxs) override {
        common_speculative_preverified result;
        if (active_tree_verify.empty() || row_idxs.size() != active_tree_verify.tokens.size()) {
            return result;
        }

        llama_synchronize(ctx);
        const float * logits = llama_get_logits(ctx);
        if (logits == nullptr) {
            active_tree_verify = {};
            return result;
        }

        const llama_model * model = llama_get_model(ctx);
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

        std::vector<llama_token> greedy_by_tree_row(active_tree_verify.tokens.size(), LLAMA_TOKEN_NULL);
        auto greedy_tree_row = [&](int32_t tree_row) -> llama_token {
            if (tree_row < 0 || tree_row >= (int32_t) greedy_by_tree_row.size()) {
                return LLAMA_TOKEN_NULL;
            }
            llama_token & cached = greedy_by_tree_row[tree_row];
            if (cached != LLAMA_TOKEN_NULL) {
                return cached;
            }

            const int32_t row = row_idxs[tree_row];
            const float * rptr = logits + (size_t) row * n_vocab;
            llama_token best = 0;
            float best_p = rptr[0];
            for (llama_token tok = 1; tok < n_vocab; ++tok) {
                if (rptr[tok] > best_p) { best_p = rptr[tok]; best = tok; }
            }
            cached = best;
            return cached;
        };

        // Fast path for single linear path (no branching) — avoids trie allocation
        if (active_tree_verify.paths.size() == 1) {
            const auto & path = active_tree_verify.paths[0];
            int32_t cur_row = 0;
            int32_t accepted_depth = 0;
            for (int32_t k = 0; k < (int32_t) path.size(); ++k) {
                if (greedy_tree_row(cur_row) != path[k]) {
                    break;
                }
                cur_row = k + 1;
                accepted_depth++;
            }

            tree_verify_result best;
            best.accepted_depth = accepted_depth;
            best.next_token     = greedy_tree_row(cur_row);
            best.hidden_row     = row_idxs[cur_row];
            best.path.assign(path.begin(), path.begin() + accepted_depth);
            best.row_path.resize(accepted_depth + 1);
            for (int32_t i = 0; i <= accepted_depth; ++i) {
                best.row_path[i] = i;
            }

            if (active_tree_verify.in_place && !best.row_path.empty()) {
                result.accepted = best.path;
                result.next     = best.next_token;
                forced_tgt_src_row = best.hidden_row;
                if (tree_timing) {
                    LOG_INF("%s: scheduler in-place linear accepted_depth=%d next=%d rows=%zu\n",
                            __func__, best.accepted_depth, best.next_token, best.row_path.size());
                }
            } else if (tree_commit && !best.row_path.empty()) {
                const int64_t t_commit_start = tree_timing ? ggml_time_us() : 0;
                if (llama_commit_mtp_tree_verify(ctx, best.row_path.data(), (int32_t) best.row_path.size())) {
                    result.accepted = best.path;
                    result.next     = best.next_token;
                    forced_tgt_src_row = best.hidden_row;
                    if (tree_timing) {
                        LOG_INF("%s: scheduler packed-committed linear accepted_depth=%d next=%d rows=%zu\n",
                                __func__, best.accepted_depth, best.next_token, best.row_path.size());
                    }
                } else {
                    LOG_WRN("%s: failed to scheduler packed-commit linear accepted_depth=%d\n",
                            __func__, best.accepted_depth);
                }
                if (t_commit_start != 0) {
                    t_tree_commit_ms += (ggml_time_us() - t_commit_start) / 1000.0;
                }
            }

            active_tree_verify = {};
            return result;
        }

        const auto nodes = build_trie_nodes(active_tree_verify.tokens[0], active_tree_verify.paths);
        if (nodes.size() != active_tree_verify.tokens.size()) {
            LOG_WRN("%s: active scheduler tree node shape changed (%zu != %zu)\n",
                    __func__, nodes.size(), active_tree_verify.tokens.size());
            active_tree_verify = {};
            return result;
        }

        tree_verify_result best;
        best.accepted_depth = 0;
        best.next_token     = greedy_tree_row(0);
        best.hidden_row     = row_idxs[0];
        best.row_path       = { 0 };
        for (int32_t p = 0; p < (int32_t) active_tree_verify.paths.size(); ++p) {
            int32_t cur = 0;
            int32_t accepted_depth = 0;
            for (int32_t k = 0; k < (int32_t) active_tree_verify.paths[p].size(); ++k) {
                const llama_token pred = greedy_tree_row(cur);
                if (pred != active_tree_verify.paths[p][k]) {
                    break;
                }
                cur = trie_child_for(nodes, cur, active_tree_verify.paths[p][k]);
                if (cur < 0) {
                    break;
                }
                accepted_depth++;
            }

            if (accepted_depth > best.accepted_depth && cur >= 0) {
                best.accepted_depth = accepted_depth;
                best.next_token     = greedy_tree_row(cur);
                best.hidden_row     = row_idxs[cur];
                best.path.assign(active_tree_verify.paths[p].begin(), active_tree_verify.paths[p].begin() + accepted_depth);
                best.row_path.clear();
                for (int32_t r = cur; r >= 0; r = nodes[r].parent) {
                    best.row_path.push_back(r);
                }
                std::reverse(best.row_path.begin(), best.row_path.end());
            }
        }

        if (active_tree_verify.in_place && !best.row_path.empty()) {
            result.accepted = best.path;
            result.next     = best.next_token;
            forced_tgt_src_row = best.hidden_row;
            if (tree_timing) {
                LOG_INF("%s: scheduler in-place branch accepted_depth=%d next=%d rows=%zu\n",
                        __func__, best.accepted_depth, best.next_token, best.row_path.size());
            }
        } else if (tree_commit && !best.row_path.empty()) {
            const int64_t t_commit_start = tree_timing ? ggml_time_us() : 0;
            if (llama_commit_mtp_tree_verify(ctx, best.row_path.data(), (int32_t) best.row_path.size())) {
                result.accepted = best.path;
                result.next     = best.next_token;
                forced_tgt_src_row = best.hidden_row;
                if (tree_timing) {
                    LOG_INF("%s: scheduler packed-committed branch accepted_depth=%d next=%d rows=%zu\n",
                            __func__, best.accepted_depth, best.next_token, best.row_path.size());
                }
            } else {
                LOG_WRN("%s: failed to scheduler packed-commit branch accepted_depth=%d\n",
                        __func__, best.accepted_depth);
            }
            if (t_commit_start != 0) {
                t_tree_commit_ms += (ggml_time_us() - t_commit_start) / 1000.0;
            }
        }

        if (result.empty()) {
            active_tree_verify = {};
        }
        return result;
    }

    void accept(uint16_t n_accepted) override {
        const int32_t n_to_drop = std::max(0, (int32_t) last_n_drafted - (int32_t) n_accepted);
        if (n_to_drop > 0) {
            rm_last_draft_tail(n_accepted);
        }
        if (invariant_checks && last_n_drafted > 0 && n_accepted > 0) {
            const uint16_t last_accepted_idx = std::min<uint16_t>(n_accepted, last_n_drafted) - 1;
            if (last_accepted_idx < last_draft_pos.size()) {
                llama_context * ctx_mtp = ctxs_mtp[multi_ctx ? std::min<size_t>(last_accepted_idx, ctxs_mtp.size() - 1) : 0];
                const llama_pos pos = last_draft_pos[last_accepted_idx];
                const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_mtp), 0);
                if (pos >= 0 && pos_max < pos) {
                    LOG_WRN("%s: MTP invariant: accepted draft idx=%d pos=%d missing after accept, pos_max=%d\n",
                            __func__, (int) last_accepted_idx, (int) pos, (int) pos_max);
                }
            }
        }
        last_n_drafted  = 0;
        last_n_accepted = (int32_t) n_accepted;
        last_draft_pos.clear();
        active_tree_verify = {};
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return std::max(1, params.draft.n_max);
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return std::max(1, params.draft.n_min);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map config;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map config)
        : common_speculative_state(type), config(std::move(config)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(config, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(config, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(config, n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_value;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_value;
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        const auto & sparams = params.ngram_mod;

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + sparams.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < sparams.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < sparams.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);
                    }

                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.ngram_mod.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.ngram_mod.n_min;
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return n_draft;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return 0;
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states

    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_MTP:           return "mtp";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    if (params.draft.model) {
        ctx_dft = llama_init_from_model(params.draft.model, params.draft.cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    auto get_mtp_layer_count = [](const llama_model * model) {
        char buf[32] = {};
        if (llama_model_meta_val_str(model, "mimo2.nextn_predict_layers", buf, sizeof(buf)) > 0) {
            return std::max(1, std::atoi(buf));
        }
        if (llama_model_meta_val_str(model, "qwen35.nextn_predict_layers", buf, sizeof(buf)) > 0) {
            return std::max(1, std::atoi(buf));
        }
        return 1;
    };

    std::vector<llama_context *> ctxs_mtp;
    if (params.has_mtp()) {
        const char * multi_ctx_env = std::getenv("LLAMA_MIMO_MTP_MULTI_CTX");
        const bool multi_ctx = multi_ctx_env != nullptr && std::strcmp(multi_ctx_env, "0") != 0;
        const int32_t n_ctxs_mtp = multi_ctx
            ? std::max(1, std::min(get_mtp_layer_count(params.mtp.model), params.draft.n_max))
            : 1;
        ctxs_mtp.reserve(n_ctxs_mtp);

        for (int32_t i = 0; i < n_ctxs_mtp; ++i) {
            llama_context * ctx_mtp = llama_init_from_model(params.mtp.model, params.mtp.cparams);
            if (ctx_mtp == nullptr) {
                LOG_ERR("%s", "failed to create MTP context\n");
                for (llama_context * ctx_mtp_prev : ctxs_mtp) {
                    llama_free(ctx_mtp_prev);
                }
                if (ctx_dft) {
                    llama_free(ctx_dft);
                }
                return nullptr;
            }
            ctxs_mtp.push_back(ctx_mtp);
        }

        LOG_INF("%s: initialized %d MTP context(s)\n", __func__, n_ctxs_mtp);
    }

    llama_context * ctx_mtp = ctxs_mtp.empty() ? nullptr : ctxs_mtp.front();

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.draft.mparams.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3
        bool has_mtp = (params.type == COMMON_SPECULATIVE_TYPE_MTP) && (ctx_mtp != nullptr);

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            auto & sparams = params.ngram_mod;

            if (!sparams.obj) {
                sparams.obj = std::make_shared<common_ngram_mod>(sparams.n_match, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                        sparams.n_match, sparams.obj->size(), (float)(sparams.obj->size_bytes())/1024/1024);

                if (sparams.n_match < 16) {
                    LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                            "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, sparams.n_match);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_MTP, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                const bool use_ckpt = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.draft.replacements,
                    /* .use_ckpt     = */ use_ckpt
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_MTP: {
                impls.push_back(std::make_unique<common_speculative_state_mtp>(
                    config.type, ctx_tgt, std::move(ctxs_mtp)));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type  = */ config.type,
                    /* .state = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config.type, config.params.ngram_map_k)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod.obj);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod.obj));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(params.ngram_cache.lookup_cache_static, params.ngram_cache.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls     = */ std::move(impls),
        /* .curr_impl = */ nullptr,
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        impl->draft_probs.clear();
        impl->draft_probs_all.clear();
        impl->draft_logits.clear();
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        {
            const int n_min = impl->n_min(params);

            if (!result.empty() && (int) result.size() < n_min) {
                LOG_DBG("%s: ignoring small draft: %d < %d\n", __func__, (int) result.size(), n_min);
                result.clear();
            }
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // we have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

common_speculative_preverified common_speculative_take_preverified(common_speculative * spec) {
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return {};
    }

    return spec->curr_impl->take_preverified();
}

common_speculative_tree_verify common_speculative_take_tree_verify(common_speculative * spec) {
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return {};
    }

    return spec->curr_impl->take_tree_verify();
}

common_speculative_preverified common_speculative_resolve_tree_verify(
        common_speculative * spec,
        llama_context * ctx,
        const std::vector<int32_t> & row_idxs) {
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return {};
    }

    return spec->curr_impl->resolve_tree_verify(ctx, row_idxs);
}

const std::vector<float> & common_speculative_get_draft_probs(const common_speculative * spec) {
    static const std::vector<float> empty;
    if (!spec || !spec->curr_impl) {
        return empty;
    }
    return spec->curr_impl->draft_probs;
}

const std::vector<std::vector<float>> & common_speculative_get_draft_logits(const common_speculative * spec) {
    static const std::vector<std::vector<float>> empty;
    if (!spec || !spec->curr_impl) {
        return empty;
    }
    return spec->curr_impl->draft_logits;
}

const std::vector<std::vector<float>> & common_speculative_get_draft_probs_all(const common_speculative * spec) {
    static const std::vector<std::vector<float>> empty;
    if (!spec || !spec->curr_impl) {
        return empty;
    }
    return spec->curr_impl->draft_probs_all;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    if (n_accepted == 0 && impl->type != COMMON_SPECULATIVE_TYPE_MTP) {
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_max = 0;
    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, impl->n_max(params));
    }

    return n_max;
}

int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_min = 0;
    for (const auto & impl : spec->impls) {
        n_min = std::max(n_min, impl->n_min(params));
    }

    return n_min;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
