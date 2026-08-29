#pragma once
#include <cstdint>
#include <string>
#include <vector>
typedef struct ggml_backend * ggml_backend_t;
struct campplus_conv {
    std::vector<float> w;
    std::vector<float> b;
    int C_out = 0, C_in = 0, k = 0;
    int kH = 0, kW = 0;
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
    int dilation_h = 1, dilation_w = 1;
    bool is_2d = false;
};
struct campplus_bn {
    std::vector<float> scale;
    std::vector<float> shift;
};
struct campplus_res_block {
    campplus_conv conv1;
    campplus_bn   bn1;
    campplus_conv conv2;
    campplus_bn   bn2;
    campplus_conv shortcut_conv;
    campplus_bn   shortcut_bn;
    int stride_h = 1;
};
struct campplus_fcm {
    campplus_conv conv1;
    campplus_bn   bn1;
    std::vector<campplus_res_block> layer1;
    std::vector<campplus_res_block> layer2;
    campplus_conv conv2;
    campplus_bn   bn2;
};
struct campplus_cam_dense_tdnn_layer {
    campplus_bn   bn1;
    campplus_conv linear1;
    campplus_bn   bn2;
    campplus_conv cam_linear_local;
    campplus_conv cam_linear1;
    campplus_conv cam_linear2;
};
struct campplus_cam_block {
    int num_layers = 0;
    int kernel_size = 3;
    int dilation = 1;
    std::vector<campplus_cam_dense_tdnn_layer> layers;
};
struct campplus_transit {
    campplus_bn   bn;
    campplus_conv linear;
};
struct campplus_weights {
    int feat_dim      = 80;
    int embedding_size= 192;
    int seg_pool_len  = 100;
    int sample_rate   = 16000;
    campplus_fcm head;
    campplus_conv tdnn_linear;
    campplus_bn   tdnn_bn;
    campplus_cam_block block1;
    campplus_transit   transit1;
    campplus_cam_block block2;
    campplus_transit   transit2;
    campplus_cam_block block3;
    campplus_transit   transit3;
    campplus_bn   out_nonlinear_bn;
    campplus_conv dense_linear;
    campplus_bn   dense_bn;
};
bool campplus_load(const std::string & s3gen_gguf_path,
                   campplus_weights & out);
bool campplus_embed(const std::vector<float>& fbank_t_by_c, int T,
                    const campplus_weights& w, std::vector<float>& out);
