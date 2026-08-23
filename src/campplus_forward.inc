


struct ve_weights_t {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
};

struct ve_bn_t {
    ggml_tensor * scale = nullptr;
    ggml_tensor * shift = nullptr;
};

struct ve_resblock_t {
    ve_weights_t conv1;  ve_bn_t bn1;
    ve_weights_t conv2;  ve_bn_t bn2;
    ve_weights_t sc;     ve_bn_t sc_bn;
    int          stride_h = 1;
    bool         has_shortcut = false;
};

struct ve_cam_layer_t {
    ve_bn_t      bn1;
    ve_weights_t lin1;
    ve_bn_t      bn2;
    ve_weights_t loc;
    ve_weights_t lin1_cam;
    ve_weights_t lin2_cam;
};

struct ve_cam_block_t {
    int num_layers = 0;
    int kernel_size = 3;
    int dilation = 1;
    int growth = 32;
    int bn_channels = 128;
    int C_in = 0;
    std::vector<ve_cam_layer_t> layers;
};

struct ve_transit_t {
    ve_bn_t      bn;
    ve_weights_t lin;
};

struct campplus_gctx {
    ggml_backend_t        backend      = nullptr;
    bool                  owns_backend = false;
    ggml_context        * weights_ctx  = nullptr;
    ggml_backend_buffer_t weights_buf  = nullptr;
    ggml_gallocr_t        allocr       = nullptr;


    int feat_dim      = 80;
    int init_channels = 128;
    int growth_rate   = 32;
    int bn_channels   = 128;
    int fcm_out_ch    = 320;
    int seg_pool_len  = 100;


    ve_weights_t fcm_conv1;  ve_bn_t fcm_bn1;
    ve_resblock_t fcm_layer1[2];
    ve_resblock_t fcm_layer2[2];
    ve_weights_t fcm_conv2;  ve_bn_t fcm_bn2;


    ve_weights_t tdnn;       ve_bn_t tdnn_bn;


    ve_cam_block_t block1, block2, block3;
    ve_transit_t   transit1, transit2, transit3;


    ve_bn_t      out_bn;
    ve_weights_t dense;
    ve_bn_t      dense_bn;
};

static void campplus_gctx_free(campplus_gctx & g) {
    if (g.allocr)      { ggml_gallocr_free(g.allocr);              g.allocr = nullptr; }
    if (g.weights_buf) { ggml_backend_buffer_free(g.weights_buf);  g.weights_buf = nullptr; }
    if (g.weights_ctx) { ggml_free(g.weights_ctx);                 g.weights_ctx = nullptr; }
    if (g.backend && g.owns_backend) ggml_backend_free(g.backend);
    g.backend = nullptr;
}


static ggml_tensor * new_f32(ggml_context * ctx, int ne0, int ne1 = 0, int ne2 = 0, int ne3 = 0) {
    if (ne3 > 0) return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    if (ne2 > 0) return ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    if (ne1 > 0) return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
}


static void decl_bn(ggml_context * ctx, ve_bn_t & b, int C, const std::string & name) {
    b.scale = new_f32(ctx, 1, C);  ggml_set_name(b.scale, (name + "/s").c_str());
    b.shift = new_f32(ctx, 1, C);  ggml_set_name(b.shift, (name + "/b").c_str());
}


static void decl_conv1d(ggml_context * ctx, ve_weights_t & w, int k, int C_in, int C_out,
                        const std::string & name, bool has_bias = false)
{
    w.w = new_f32(ctx, k, C_in, C_out);  ggml_set_name(w.w, (name + "/w").c_str());
    if (has_bias) {

        w.b = new_f32(ctx, 1, C_out);  ggml_set_name(w.b, (name + "/b").c_str());
    }
}


static void decl_conv2d(ggml_context * ctx, ve_weights_t & w, int kW, int kH, int C_in, int C_out,
                        const std::string & name)
{
    w.w = new_f32(ctx, kW, kH, C_in, C_out);
    ggml_set_name(w.w, (name + "/w").c_str());
}


static void upload(ggml_tensor * t, const void * src, size_t bytes) {
    ggml_backend_tensor_set(t, src, 0, bytes);
}

static void upload_bn(const ve_bn_t & bn, const campplus_bn & src) {
    upload(bn.scale, src.scale.data(), src.scale.size() * sizeof(float));
    upload(bn.shift, src.shift.data(), src.shift.size() * sizeof(float));
}


static bool init_weight(const ve_weights_t & w, const campplus_conv & src,
                        bool expect_bias = false)
{
    const size_t want = src.w.size() * sizeof(float);
    if (ggml_nbytes(w.w) != want) {
        fprintf(stderr, "campplus_ggml: weight size mismatch for %s: have %zu, want %zu\n",
                ggml_get_name(w.w), ggml_nbytes(w.w), want);
        return false;
    }
    upload(w.w, src.w.data(), want);
    if (expect_bias) {
        if (!w.b || src.b.empty()) {
            fprintf(stderr, "campplus_ggml: missing bias for %s\n", ggml_get_name(w.w));
            return false;
        }
        upload(w.b, src.b.data(), src.b.size() * sizeof(float));
    }
    return true;
}


static ggml_tensor * bn_apply_1d(ggml_context * ctx, ggml_tensor * x, const ve_bn_t & bn) {
    x = ggml_mul(ctx, x, bn.scale);
    x = ggml_add(ctx, x, bn.shift);
    return x;
}


static ggml_tensor * bn_apply_2d(ggml_context * ctx, ggml_tensor * x, const ve_bn_t & bn) {
    const int C = (int) bn.scale->ne[1];
    ggml_tensor * s = ggml_reshape_4d(ctx, bn.scale, 1, 1, C, 1);
    ggml_tensor * b = ggml_reshape_4d(ctx, bn.shift, 1, 1, C, 1);
    x = ggml_mul(ctx, x, s);
    x = ggml_add(ctx, x, b);
    return x;
}


static ggml_tensor * conv1d_f32(ggml_context * ctx,
                                ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation)
{
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input,
                                       stride, 0, padding, 0, dilation, 0,
                                       false, GGML_TYPE_F32);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, r, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}


static ggml_tensor * conv2d_f32(ggml_context * ctx,
                                ggml_tensor * kernel, ggml_tensor * input,
                                int sW, int sH, int pW, int pH)
{

    return ggml_conv_2d(ctx, kernel, input, sW, sH, pW, pH, 1, 1);
}

static ggml_tensor * fcm_res_block(ggml_context * ctx, ggml_tensor * x,
                                   const ve_resblock_t & blk)
{
    ggml_tensor * t1 = conv2d_f32(ctx, blk.conv1.w, x, 1, blk.stride_h, 1, 1);
    t1 = bn_apply_2d(ctx, t1, blk.bn1);
    t1 = ggml_relu(ctx, t1);

    ggml_tensor * t2 = conv2d_f32(ctx, blk.conv2.w, t1, 1, 1, 1, 1);
    t2 = bn_apply_2d(ctx, t2, blk.bn2);

    ggml_tensor * sc = x;
    if (blk.has_shortcut) {
        sc = conv2d_f32(ctx, blk.sc.w, x, 1, blk.stride_h, 0, 0);
        sc = bn_apply_2d(ctx, sc, blk.sc_bn);
    }
    ggml_tensor * y = ggml_add(ctx, t2, sc);
    y = ggml_relu(ctx, y);
    return y;
}


static ggml_tensor * cam_dense_tdnn_layer(ggml_context * ctx, ggml_tensor * x,
                                          const ve_cam_layer_t & L,
                                          int kernel_size, int dilation, int seg_pool_len,
                                          int growth, int bn_channels,
                                          ggml_tensor * seg_scale,
                                          int T_pad, int S, int T)
{

    ggml_tensor * y = bn_apply_1d(ctx, x, L.bn1);
    y = ggml_relu(ctx, y);


    ggml_tensor * z = conv1d_f32(ctx, L.lin1.w, y, 1, 0, 1);


    z = bn_apply_1d(ctx, z, L.bn2);
    z = ggml_relu(ctx, z);


    const int pad = (kernel_size - 1) / 2 * dilation;
    ggml_tensor * y_local = conv1d_f32(ctx, L.loc.w, z, 1, pad, dilation);


    ggml_tensor * mean_ctx = ggml_mean(ctx, z);


    const int64_t C_bn = z->ne[1];
    ggml_tensor * z_pad = z;
    if (T_pad > T) {
        z_pad = ggml_pad(ctx, z, T_pad - T, 0, 0, 0);
    }
    ggml_tensor * z_r     = ggml_reshape_3d(ctx, z_pad, seg_pool_len, S, C_bn);
    ggml_tensor * seg_sum = ggml_sum_rows (ctx, z_r);
    ggml_tensor * seg_mean = ggml_mul(ctx, seg_sum, seg_scale);
    ggml_tensor * seg_bc  = ggml_repeat (ctx, seg_mean, z_r);
    ggml_tensor * seg_flat = ggml_reshape_2d(ctx, seg_bc, T_pad, C_bn);
    ggml_tensor * seg_ctx;
    if (T_pad > T) {
        seg_ctx = ggml_view_2d(ctx, seg_flat, T, C_bn, seg_flat->nb[1], 0);
    } else {
        seg_ctx = seg_flat;
    }


    ggml_tensor * context = ggml_add(ctx, seg_ctx, mean_ctx);


    ggml_tensor * h1 = conv1d_f32(ctx, L.lin1_cam.w, context, 1, 0, 1);
    h1 = ggml_add(ctx, h1, L.lin1_cam.b);
    h1 = ggml_relu(ctx, h1);


    ggml_tensor * gate = conv1d_f32(ctx, L.lin2_cam.w, h1, 1, 0, 1);
    gate = ggml_add(ctx, gate, L.lin2_cam.b);
    gate = ggml_sigmoid(ctx, gate);

    ggml_tensor * cam_out = ggml_mul(ctx, y_local, gate);


    return ggml_concat(ctx, x, cam_out, 1);
}

static ggml_tensor * build_cam_block(ggml_context * ctx, ggml_tensor * x,
                                     const ve_cam_block_t & blk, int seg_pool_len,
                                     ggml_tensor * seg_scale,
                                     int T_pad, int S, int T)
{
    for (int i = 0; i < blk.num_layers; ++i) {
        x = cam_dense_tdnn_layer(ctx, x, blk.layers[i],
                                  blk.kernel_size, blk.dilation, seg_pool_len,
                                  blk.growth, blk.bn_channels,
                                  seg_scale, T_pad, S, T);
    }
    return x;
}

static ggml_tensor * build_transit(ggml_context * ctx, ggml_tensor * x,
                                   const ve_transit_t & t)
{
    x = bn_apply_1d(ctx, x, t.bn);
    x = ggml_relu(ctx, x);
    x = conv1d_f32(ctx, t.lin.w, x, 1, 0, 1);
    return x;
}


static ggml_tensor * build_stats_pool(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * mean = ggml_mean(ctx, x);
    ggml_tensor * diff = ggml_sub(ctx, x, mean);
    ggml_tensor * sq   = ggml_mul(ctx, diff, diff);
    ggml_tensor * var  = ggml_mean(ctx, sq);
    ggml_tensor * std_ = ggml_sqrt(ctx, var);
    ggml_tensor * cat  = ggml_concat(ctx, mean, std_, 1);
    return cat;
}


static ggml_cgraph * build_campplus_graph(const campplus_gctx & g, int T) {
    const int max_nodes = 4096;
    const size_t buf_size = ggml_tensor_overhead() * max_nodes +
                            ggml_graph_overhead_custom(max_nodes, false);
    static std::vector<uint8_t> buf;
    buf.resize(buf_size);
    ggml_init_params p = { buf_size, buf.data(),  true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf   = ggml_new_graph_custom(ctx, max_nodes, false);


    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, g.feat_dim, 1, 1);
    ggml_set_name(x, "fbank"); ggml_set_input(x);


    ggml_tensor * y = conv2d_f32(ctx, g.fcm_conv1.w, x, 1, 1, 1, 1);
    y = bn_apply_2d(ctx, y, g.fcm_bn1);
    y = ggml_relu(ctx, y);

    for (int i = 0; i < 2; ++i) y = fcm_res_block(ctx, y, g.fcm_layer1[i]);
    for (int i = 0; i < 2; ++i) y = fcm_res_block(ctx, y, g.fcm_layer2[i]);


    y = conv2d_f32(ctx, g.fcm_conv2.w, y, 1, 2, 1, 1);
    y = bn_apply_2d(ctx, y, g.fcm_bn2);
    y = ggml_relu(ctx, y);


    ggml_tensor * y_cont = ggml_cont(ctx, y);
    ggml_tensor * x1 = ggml_reshape_2d(ctx, y_cont, y_cont->ne[0],
                                        y_cont->ne[1] * y_cont->ne[2]);


    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);


    x1 = conv1d_f32(ctx, g.tdnn.w, x1, 2, 2, 1);

    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d(ctx, x1, g.tdnn_bn);
    x1 = ggml_relu(ctx, x1);


    const int T_cam   = (int) x1->ne[0];
    const int S       = (T_cam + g.seg_pool_len - 1) / g.seg_pool_len;
    const int T_pad   = S * g.seg_pool_len;


    ggml_tensor * seg_scale = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, S, 1);
    ggml_set_name(seg_scale, "seg_scale"); ggml_set_input(seg_scale);


    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);


    x1 = build_cam_block (ctx, x1, g.block1, g.seg_pool_len, seg_scale, T_pad, S, T_cam);
    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d    (ctx, x1, g.transit1.bn);
    x1 = ggml_relu      (ctx, x1);
    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);
    x1 = conv1d_f32     (ctx, g.transit1.lin.w, x1, 1, 0, 1);

    x1 = build_cam_block (ctx, x1, g.block2, g.seg_pool_len, seg_scale, T_pad, S, T_cam);
    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d    (ctx, x1, g.transit2.bn);
    x1 = ggml_relu      (ctx, x1);
    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);
    x1 = conv1d_f32     (ctx, g.transit2.lin.w, x1, 1, 0, 1);

    x1 = build_cam_block (ctx, x1, g.block3, g.seg_pool_len, seg_scale, T_pad, S, T_cam);
    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d    (ctx, x1, g.transit3.bn);
    x1 = ggml_relu      (ctx, x1);
    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);
    x1 = conv1d_f32     (ctx, g.transit3.lin.w, x1, 1, 0, 1);


    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d    (ctx, x1, g.out_bn);
    x1 = ggml_relu      (ctx, x1);


    ggml_tensor * stats = build_stats_pool(ctx, x1);


    ggml_tensor * stats_row = ggml_reshape_2d(ctx, stats, stats->ne[1], 1);

    ggml_tensor * dense_w2 = ggml_reshape_2d(ctx, g.dense.w,
                                               g.dense.w->ne[1], g.dense.w->ne[2]);
    ggml_tensor * emb = ggml_mul_mat(ctx, dense_w2, stats_row);
    emb = ggml_reshape_2d(ctx, emb, 1, emb->ne[0]);
    emb = bn_apply_1d(ctx, emb, g.dense_bn);

    ggml_set_name(emb, "emb"); ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    ggml_free(ctx);
    return gf;
}


static bool upload_cam_layer(const ve_cam_layer_t & gl, const campplus_cam_dense_tdnn_layer & src) {
    upload_bn(gl.bn1, src.bn1);
    if (!init_weight(gl.lin1,  src.linear1))                    return false;
    upload_bn(gl.bn2, src.bn2);
    if (!init_weight(gl.loc,   src.cam_linear_local))           return false;
    if (!init_weight(gl.lin1_cam, src.cam_linear1, true))       return false;
    if (!init_weight(gl.lin2_cam, src.cam_linear2, true))       return false;
    return true;
}

static bool upload_weights(campplus_gctx & g, const campplus_weights & w) {

    if (!init_weight(g.fcm_conv1, w.head.conv1)) return false;
    upload_bn(g.fcm_bn1, w.head.bn1);
    for (int i = 0; i < 2; ++i) {
        const auto & src = w.head.layer1[i];
        auto & dst = g.fcm_layer1[i];
        if (!init_weight(dst.conv1, src.conv1)) return false;
        upload_bn(dst.bn1, src.bn1);
        if (!init_weight(dst.conv2, src.conv2)) return false;
        upload_bn(dst.bn2, src.bn2);
        if (dst.has_shortcut) {
            if (!init_weight(dst.sc, src.shortcut_conv)) return false;
            upload_bn(dst.sc_bn, src.shortcut_bn);
        }
    }
    for (int i = 0; i < 2; ++i) {
        const auto & src = w.head.layer2[i];
        auto & dst = g.fcm_layer2[i];
        if (!init_weight(dst.conv1, src.conv1)) return false;
        upload_bn(dst.bn1, src.bn1);
        if (!init_weight(dst.conv2, src.conv2)) return false;
        upload_bn(dst.bn2, src.bn2);
        if (dst.has_shortcut) {
            if (!init_weight(dst.sc, src.shortcut_conv)) return false;
            upload_bn(dst.sc_bn, src.shortcut_bn);
        }
    }
    if (!init_weight(g.fcm_conv2, w.head.conv2)) return false;
    upload_bn(g.fcm_bn2, w.head.bn2);


    if (!init_weight(g.tdnn, w.tdnn_linear)) return false;
    upload_bn(g.tdnn_bn, w.tdnn_bn);


    auto upload_block = [&](ve_cam_block_t & gb, const campplus_cam_block & sb) -> bool {
        for (int i = 0; i < gb.num_layers; ++i) {
            if (!upload_cam_layer(gb.layers[i], sb.layers[i])) return false;
        }
        return true;
    };
    if (!upload_block(g.block1, w.block1)) return false;
    upload_bn(g.transit1.bn, w.transit1.bn);
    if (!init_weight(g.transit1.lin, w.transit1.linear)) return false;
    if (!upload_block(g.block2, w.block2)) return false;
    upload_bn(g.transit2.bn, w.transit2.bn);
    if (!init_weight(g.transit2.lin, w.transit2.linear)) return false;
    if (!upload_block(g.block3, w.block3)) return false;
    upload_bn(g.transit3.bn, w.transit3.bn);
    if (!init_weight(g.transit3.lin, w.transit3.linear)) return false;

    upload_bn(g.out_bn, w.out_nonlinear_bn);
    if (!init_weight(g.dense, w.dense_linear)) return false;
    upload_bn(g.dense_bn, w.dense_bn);
    return true;
}


static bool declare_weights(campplus_gctx & g, const campplus_weights & w) {


    const int n_tensors = 2000;
    ggml_init_params ip = {
         (size_t) n_tensors * ggml_tensor_overhead(),
         nullptr,
         true,
    };
    g.weights_ctx = ggml_init(ip);
    if (!g.weights_ctx) {
        fprintf(stderr, "campplus_ggml: ggml_init failed\n"); return false;
    }

    g.feat_dim      = w.feat_dim;
    g.seg_pool_len  = w.seg_pool_len;
    g.fcm_out_ch    = 32 * (w.feat_dim / 8);
    g.init_channels = w.tdnn_linear.C_out;
    g.growth_rate   = 32;
    g.bn_channels   = 128;


    decl_conv2d(g.weights_ctx, g.fcm_conv1, 3, 3, 1, 32, "fcm_conv1");
    decl_bn    (g.weights_ctx, g.fcm_bn1, 32, "fcm_bn1");

    auto setup_rb = [&](ve_resblock_t & rb, const campplus_res_block & src,
                        int in_ch, int planes, int stride, const std::string & name) {
        rb.stride_h    = stride;
        rb.has_shortcut = !src.shortcut_conv.w.empty();
        decl_conv2d(g.weights_ctx, rb.conv1, 3, 3, in_ch, planes, name + "_conv1");
        decl_bn    (g.weights_ctx, rb.bn1, planes, name + "_bn1");
        decl_conv2d(g.weights_ctx, rb.conv2, 3, 3, planes, planes, name + "_conv2");
        decl_bn    (g.weights_ctx, rb.bn2, planes, name + "_bn2");
        if (rb.has_shortcut) {
            decl_conv2d(g.weights_ctx, rb.sc, 1, 1, in_ch, planes, name + "_sc");
            decl_bn    (g.weights_ctx, rb.sc_bn, planes, name + "_scbn");
        }
    };
    setup_rb(g.fcm_layer1[0], w.head.layer1[0], 32, 32, 2, "fcm_l10");
    setup_rb(g.fcm_layer1[1], w.head.layer1[1], 32, 32, 1, "fcm_l11");
    setup_rb(g.fcm_layer2[0], w.head.layer2[0], 32, 32, 2, "fcm_l20");
    setup_rb(g.fcm_layer2[1], w.head.layer2[1], 32, 32, 1, "fcm_l21");

    decl_conv2d(g.weights_ctx, g.fcm_conv2, 3, 3, 32, 32, "fcm_conv2");
    decl_bn    (g.weights_ctx, g.fcm_bn2, 32, "fcm_bn2");


    decl_conv1d(g.weights_ctx, g.tdnn, 5, g.fcm_out_ch, g.init_channels, "tdnn");
    decl_bn    (g.weights_ctx, g.tdnn_bn, g.init_channels, "tdnn_bn");


    auto setup_block = [&](ve_cam_block_t & gb, const campplus_cam_block & sb,
                           int C_in, const std::string & prefix) {
        gb.num_layers  = sb.num_layers;
        gb.kernel_size = sb.kernel_size;
        gb.dilation    = sb.dilation;
        gb.growth      = g.growth_rate;
        gb.bn_channels = g.bn_channels;
        gb.C_in        = C_in;
        gb.layers.resize(gb.num_layers);
        for (int i = 0; i < gb.num_layers; ++i) {
            const int layer_C_in = C_in + i * g.growth_rate;
            auto & L = gb.layers[i];
            const std::string name = prefix + "_l" + std::to_string(i);
            decl_bn    (g.weights_ctx, L.bn1, layer_C_in, name + "_bn1");
            decl_conv1d(g.weights_ctx, L.lin1, 1, layer_C_in, g.bn_channels, name + "_lin1");
            decl_bn    (g.weights_ctx, L.bn2, g.bn_channels, name + "_bn2");
            decl_conv1d(g.weights_ctx, L.loc, gb.kernel_size, g.bn_channels, g.growth_rate, name + "_loc");
            decl_conv1d(g.weights_ctx, L.lin1_cam, 1, g.bn_channels, g.bn_channels / 2, name + "_cam1", true);
            decl_conv1d(g.weights_ctx, L.lin2_cam, 1, g.bn_channels / 2, g.growth_rate,  name + "_cam2", true);
        }
    };
    setup_block(g.block1, w.block1, g.init_channels, "b1");
    const int after_b1 = g.init_channels + g.block1.num_layers * g.growth_rate;
    decl_bn    (g.weights_ctx, g.transit1.bn, after_b1, "tr1_bn");
    decl_conv1d(g.weights_ctx, g.transit1.lin, 1, after_b1, after_b1 / 2, "tr1_lin");

    const int b2_in = after_b1 / 2;
    setup_block(g.block2, w.block2, b2_in, "b2");
    const int after_b2 = b2_in + g.block2.num_layers * g.growth_rate;
    decl_bn    (g.weights_ctx, g.transit2.bn, after_b2, "tr2_bn");
    decl_conv1d(g.weights_ctx, g.transit2.lin, 1, after_b2, after_b2 / 2, "tr2_lin");

    const int b3_in = after_b2 / 2;
    setup_block(g.block3, w.block3, b3_in, "b3");
    const int after_b3 = b3_in + g.block3.num_layers * g.growth_rate;
    decl_bn    (g.weights_ctx, g.transit3.bn, after_b3, "tr3_bn");
    decl_conv1d(g.weights_ctx, g.transit3.lin, 1, after_b3, after_b3 / 2, "tr3_lin");

    const int final_ch = after_b3 / 2;
    decl_bn    (g.weights_ctx, g.out_bn, final_ch, "out_bn");
    decl_conv1d(g.weights_ctx, g.dense, 1, final_ch * 2, w.embedding_size, "dense");
    decl_bn    (g.weights_ctx, g.dense_bn, w.embedding_size, "dense_bn");

    g.weights_buf = ggml_backend_alloc_ctx_tensors(g.weights_ctx, g.backend);
    if (!g.weights_buf) {
        fprintf(stderr, "campplus_ggml: weight buffer alloc failed\n"); return false;
    }
    return true;
}


static bool campplus_embed_ggml(const std::vector<float> & fbank_t_by_c, int T,
                                const campplus_weights & w, ggml_backend_t backend,
                                std::vector<float> & out)
{
    if ((int64_t) fbank_t_by_c.size() != (int64_t) T * w.feat_dim) {
        fprintf(stderr, "campplus_embed_ggml: fbank has %zu elts, expected %d*%d\n",
                fbank_t_by_c.size(), T, w.feat_dim);
        return false;
    }


    campplus_gctx g;
    g.backend = backend;
    if (!g.backend) {
        g.backend      = ggml_backend_cpu_init();
        g.owns_backend = true;
        if (!g.backend) { fprintf(stderr, "campplus_ggml: cpu init failed\n"); return false; }
    }

    if (!declare_weights(g, w)) { campplus_gctx_free(g); return false; }
    if (!upload_weights (g, w)) { campplus_gctx_free(g); return false; }

    ggml_cgraph * gf = build_campplus_graph(g, T);

    g.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g.backend));
    if (!g.allocr || !ggml_gallocr_reserve(g.allocr, gf)) {
        fprintf(stderr, "campplus_ggml: gallocr reserve failed\n");
        campplus_gctx_free(g); return false;
    }
    ggml_gallocr_alloc_graph(g.allocr, gf);


    ggml_tensor * fb = ggml_graph_get_tensor(gf, "fbank");
    upload(fb, fbank_t_by_c.data(), fbank_t_by_c.size() * sizeof(float));


    {
        const int T_cam = (T + 1) / 2;
        const int S     = (T_cam + w.seg_pool_len - 1) / w.seg_pool_len;
        const int T_pad = S * w.seg_pool_len;
        std::vector<float> scale((size_t) S);
        for (int s = 0; s < S - 1; ++s) scale[s] = 1.0f / (float) w.seg_pool_len;
        scale[S - 1] = 1.0f / (float) (T_cam - (S - 1) * w.seg_pool_len);
        (void) T_pad;
        ggml_tensor * ss = ggml_graph_get_tensor(gf, "seg_scale");
        if (ss) upload(ss, scale.data(), scale.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(g.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "campplus_ggml: graph_compute failed\n");
        campplus_gctx_free(g); return false;
    }

    ggml_tensor * emb = ggml_graph_get_tensor(gf, "emb");
    out.assign((size_t) w.embedding_size, 0.0f);
    ggml_backend_tensor_get(emb, out.data(), 0, out.size() * sizeof(float));

    campplus_gctx_free(g);
    return true;
}
