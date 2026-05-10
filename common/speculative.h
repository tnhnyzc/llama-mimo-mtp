#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

struct common_speculative_preverified {
    llama_tokens accepted;
    llama_token  next = LLAMA_TOKEN_NULL;

    bool empty() const {
        return accepted.empty() && next == LLAMA_TOKEN_NULL;
    }
};

struct common_speculative_tree_verify {
    llama_tokens tokens;
    std::vector<int32_t> parents;
    std::vector<int32_t> depths;
    std::vector<llama_tokens> paths;

    int32_t n_steps = 0;
    int32_t top_k   = 0;
    bool    in_place = false;
    bool    packed   = false;

    bool empty() const {
        return tokens.empty();
    }
};

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// optional fast path: returns accepted tokens plus the next sampled token from an already-verified target path
common_speculative_preverified common_speculative_take_preverified(common_speculative * spec);

// optional packed target verification path: returns a tree batch prepared by the speculative decoder
common_speculative_tree_verify common_speculative_take_tree_verify(common_speculative * spec);

// resolves a previously decoded packed tree batch and returns accepted tokens plus the next sampled token
common_speculative_preverified common_speculative_resolve_tree_verify(
        common_speculative                         * spec,
        struct llama_context                      * ctx,
        const std::vector<int32_t>                & row_idxs);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// returns the draft probabilities q(x) for the most recent draft, for p/q acceptance
const std::vector<float> & common_speculative_get_draft_probs(const common_speculative * spec);

// returns raw draft logits per position for logit blending
const std::vector<std::vector<float>> & common_speculative_get_draft_logits(const common_speculative * spec);

// returns full draft softmax distribution per position for rejection resampling
const std::vector<std::vector<float>> & common_speculative_get_draft_probs_all(const common_speculative * spec);

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params);
int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
