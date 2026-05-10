#include "models.h"

#include <cstdlib>
#include <cstring>

void llama_model_mimo2_mtp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,   hparams.n_swa);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,         hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE,      hparams.f_attn_value_scale, false);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,       hparams.nextn_predict_layers, false);
    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "MIMO2_MTP requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.nextn_predict_layers <= hparams.n_layer);

    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
    const uint32_t n_main = hparams.n_layer - hparams.nextn_predict_layers;
    uint32_t n_head_kv_swa = hparams.n_head_kv_arr[n_main > 0 ? n_main - 1 : 0];
    for (uint32_t i = 0; i < n_main; ++i) {
        if (hparams.swa_layers[i]) {
            n_head_kv_swa = std::max(n_head_kv_swa, hparams.n_head_kv_arr[i]);
        }
    }
    for (uint32_t i = n_main; i < hparams.n_layer; ++i) {
        hparams.swa_layers[i]    = 1;
        hparams.n_head_arr[i]    = hparams.n_head_arr[0];
        hparams.n_head_kv_arr[i] = n_head_kv_swa;
        hparams.n_ff_arr[i]      = hparams.n_ff_arr[0];
    }

    hparams.kv_only_nextn         = true;
    hparams.n_layer_kv_from_start = -1;
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = false;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_mimo2_mtp::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    const uint32_t n_main = n_layer - hparams.nextn_predict_layers;
    for (int i = 0; i < n_layer; ++i) {
        if (static_cast<uint32_t>(i) < n_main) {
            continue;
        }

        auto & layer = layers[i];

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i);
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i);
        const uint32_t n_head       = hparams.n_head(i);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT,   "weight", i), {n_embd_head_v * n_head, n_embd}, 0);
        layer.attn_norm  = create_tensor(tn(LLM_TENSOR_ATTN_NORM,  "weight", i), {n_embd}, 0);
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);

        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, 0);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd},             0);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd},             0);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd},             TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_mimo2_mtp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_mimo2_mtp::graph::graph(const llama_model & model, const llm_graph_params & params)
        : llm_graph_context(params) {
    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "MIMO2_MTP requires nextn_predict_layers > 0");

    const int n_main = (int) hparams.n_layer - (int) hparams.nextn_predict_layers;
    int mtp_idx = params.mtp_layer_idx < 0 ? 0 : params.mtp_layer_idx;
    const char * fixed_layer_env = std::getenv("LLAMA_MIMO_MTP_FIXED_LAYER");
    if (fixed_layer_env != nullptr) {
        mtp_idx = std::atoi(fixed_layer_env);
    }
    if (mtp_idx >= (int) hparams.nextn_predict_layers) {
        mtp_idx = (int) hparams.nextn_predict_layers - 1;
    }
    if (mtp_idx < 0) {
        mtp_idx = 0;
    }

    const int il = n_main + mtp_idx;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MIMO2_MTP missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MIMO2_MTP missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MIMO2_MTP missing nextn.hnorm");
    GGML_ASSERT(layer.wqkv          && "MIMO2_MTP requires fused attn_qkv");

    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * h_input    = inp->embd;
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn       = build_attn_inp_kv_iswa();

    ggml_tensor * h_norm = build_norm(h_input, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    const uint32_t n_head_l       = hparams.n_head(il);
    const uint32_t n_head_kv_l    = hparams.n_head_kv(il);
    const float    freq_base_l    = model.get_rope_freq_base(cparams, il);
    const float    freq_scale_l   = model.get_rope_freq_scale(cparams, il);
    const float    v_scale        = hparams.f_attn_value_scale;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur);
    cb(qkv, "mtp_wqkv", il);

    const size_t row_k    = ggml_row_size(qkv->type, n_embd_head_k);
    const size_t row_v    = ggml_row_size(qkv->type, n_embd_head_v);
    const size_t row_full = qkv->nb[1];
    const size_t k_off    = row_k * n_head_l;
    const size_t v_off    = k_off + row_k * n_head_kv_l;

    ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_l,    n_tokens, row_k, row_full, 0);
    ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_kv_l, n_tokens, row_k, row_full, k_off);
    ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_kv_l, n_tokens, row_v, row_full, v_off);

    Qcur = ggml_rope_ext(
        ctx0, Qcur, inp_pos, nullptr,
        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
        ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_ext(
        ctx0, Kcur, inp_pos, nullptr,
        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
        ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);
    cb(Vcur, "mtp_Vcur", il);

    cur = build_attn(inp_attn,
            layer.wo, nullptr, layer.wo_s,
            Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr,
            1.0f / sqrtf(float(n_embd_head_k)), il);
    cb(cur, "mtp_attn_out", il);

    if (v_scale != 1.0f) {
        cur = ggml_scale(ctx0, cur, v_scale);
        cb(cur, "mtp_attn_out_scaled", il);
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    // Qwen/Step feed the next AR MTP step with the post-block hidden before
    // the logits head norm. MTPLX's MiMo-V2 probe keeps the post-final-norm
    // variant plausible, so leave a cheap runtime switch for acceptance tests.
    const char * post_norm_hidden_env = std::getenv("LLAMA_MIMO_MTP_POST_NORM_HIDDEN");
    const bool use_post_norm_hidden = post_norm_hidden_env != nullptr && std::strcmp(post_norm_hidden_env, "0") != 0;
    if (!use_post_norm_hidden) {
        res->t_mtp_out = cur;
    }

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm_w && "MIMO2_MTP missing shared head norm fallback");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);
    if (use_post_norm_hidden) {
        res->t_mtp_out = cur;
    }

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "MIMO2_MTP missing LM head fallback");
    cur = build_lora_mm(head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
