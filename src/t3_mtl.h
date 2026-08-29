#pragma once
#include "chatterbox_t3_internal.h"
#include "ggml.h"
#include "ggml-backend.h"
namespace tts_cpp::chatterbox::detail {
void t3_stack_unregister(ggml_backend_buffer_t buf, ggml_context * ctx);
ggml_cgraph * build_stage_cond_emb_graph(const chatterbox_model & m);
ggml_cgraph * build_stage_text_emb_graph(const chatterbox_model & m, int T_text);
ggml_cgraph * build_stage_inputs_graph(const chatterbox_model & m, int T_text,
                                       bool is_uncond);
ggml_cgraph * build_stage_layers_graph(const chatterbox_model & m, int N,
                                       int n_layers, bool is_uncond);
ggml_cgraph * build_stage_head_graph(const chatterbox_model & m, int N);
}
