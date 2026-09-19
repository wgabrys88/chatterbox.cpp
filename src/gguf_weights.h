#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstring>
#include <string>
#include <vector>

struct GgufWeights {
    ggml_context * ctx = nullptr;
    gguf_context * file;
    explicit GgufWeights(const std::string& path) {
        file = gguf_init_from_file(path.c_str(), {false, &ctx});
    }
    ~GgufWeights() {
        gguf_free(file);
        ggml_free(ctx);
    }
    uint32_t u32(const char* key) const { return gguf_get_val_u32(file, gguf_find_key(file, key)); }
    float f32(const char* key) const { return gguf_get_val_f32(file, gguf_find_key(file, key)); }
    void copy(const char* name, std::vector<float>& out) const {
        copy_tensor(ctx, name, out);
    }
    static void copy_tensor(ggml_context* ctx, const char* name, std::vector<float>& out) {
        auto* tensor = ggml_get_tensor(ctx, name);
        out.resize(ggml_nelements(tensor));
        std::memcpy(out.data(), ggml_get_data(tensor), ggml_nbytes(tensor));
    }
    template<class Model> void upload(Model& model, bool expand_conv = false) const {
        model.ctx_w = ggml_init({ggml_tensor_overhead() * size_t(gguf_get_n_tensors(file)), nullptr, true});
        for (int64_t i = 0; i < gguf_get_n_tensors(file); ++i) {
            const char* name = gguf_get_tensor_name(file, i);
            auto* src = ggml_get_tensor(ctx, name);
            auto* dst = expand_conv && src->type == GGML_TYPE_F16 && ggml_is_3d(src)
                ? ggml_new_tensor(model.ctx_w, GGML_TYPE_F32, ggml_n_dims(src), src->ne)
                : ggml_dup_tensor(model.ctx_w, src);
            ggml_set_name(dst, name);
            model.tensors[name] = dst;
        }
        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
        for (auto* dst = ggml_get_first_tensor(model.ctx_w); dst; dst = ggml_get_next_tensor(model.ctx_w, dst)) {
            auto* src = ggml_get_tensor(ctx, ggml_get_name(dst));
            if (dst->type != src->type) {
                std::vector<float> data(ggml_nelements(src));
                ggml_fp16_to_fp32_row((const ggml_fp16_t*)ggml_get_data(src), data.data(), data.size());
                ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(float));
            } else {
                ggml_backend_tensor_set(dst, ggml_get_data(src), 0, ggml_nbytes(src));
            }
        }
    }
};
