/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <vector>

// <NT> 启发式判断是否使用pack_gqa，与之相对应的是nopack_gqa。
// PackGQA 是 FlashAttention 中对 GQA 的一种优化实现，通过更紧凑的内存布局和索引机制，减少了 KV 缓存的大小，提高了计算效率。
// 
// 启发式经验：PackGQA 的速度稍慢一些，但如果seqlen_q较小，或者不是 kBlockM 的倍数附近时，它可能会有所帮助。
// 所以nopack_gqa_efficiency大于等于0.9 * pack_gqa_efficiency时就会用nopack_gqa。
inline bool should_pack_gqa(bool varlen_q, int seqlen_q, int qhead_per_khead, int blockM) {
    // If varlen, we don't actually know seqlen_q but only max_seqlen_q.
    if (varlen_q) return true;
    // Heuristic: PackGQA is a bit slower but can help if seqlen_q is small or not near a multiple of kBlockM
    // <NT> a向上取整到b的倍数，如a=11，b=3，(11+3-1)/3*3=13/3*3=12.
    auto round_up = [](int a, int b) { return (a + b - 1) / b * b; };
    float nopack_gqa_efficiency = float(seqlen_q) / float(round_up(seqlen_q, blockM));
    float pack_gqa_efficiency = float(seqlen_q * qhead_per_khead) / float(round_up(seqlen_q * qhead_per_khead, blockM));
    return nopack_gqa_efficiency < 0.9 * pack_gqa_efficiency;
};

// <NT> 启发式寻找split数量，让使用率最大化。例如batch*n_heads=48, 并有108个sm，则splits为2时，用两个sm负责一个batch*n_head
// (切的是seq长度方向，跟batch和n_heads不是一个维度，split为2，则分成了两块同步进行，batch*n_heads中的一份变成了两份，从一个sm负责变成了两个sm负责)，则使用率为48/(108/2)=0.89.
// 而splits为3时，3个sm负责一个，使用率为48/(108/3)=48/36=1.33, 超了100%，一个wave处理不完，需要两个wave来处理，所以需要48/36*2=0.667。
// 总之，公式为 float n_waves = float(total_mblocks * num_splits) / num_SMs;
//             float eff = n_waves / ceil(n_waves);
// 另外split如果太多，会导致更多HBM的读写，所以需要权衡，这里启发式搜索的基本准则是使用率能达到85%的最小的splits数量。
// 此外需要满足KV的每个head能完整填充到L2里，以免影响读取效率，这里假定L2是50MB 
// (H20的L2是60MB; H200的L2是50MB，L1是256KB/sm; L40的L2是96MB, L1是128KB/sm): https://www.techpowerup.com/gpu-specs/
// 相关博客：https://zhuanlan.zhihu.com/p/688345042

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int total_mblocks, int num_SMs, int num_n_blocks, int num_m_blocks, int size_one_kv_head, bool is_causal_or_local, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    // However, in the case of super long seqlen where each head of KV doesn't even fit into
    // L2 (we assume that L2 size is 50MB), we want to split.
    if (total_mblocks >= 0.8f * num_SMs) {
        int const size_l2 = 50 * 1024 * 1024;
        // Only split if there are enough queries to go over the KV at least twice
        // Don't split if causal
        if (size_one_kv_head > size_l2 && num_m_blocks >= num_SMs * 2 && !is_causal_or_local) {
            return std::min((size_one_kv_head + size_l2 - 1) / size_l2, max_splits);
        } else {
            return 1;
        }
    }
    // <NT> qwen2模型的head_dim是128的，deepseekv3里的标准的也是128，其中qk还会拼接上rope_dim=64; 
    //      deepseek v3中q_lora_rank=1536， kv_lora_rank=512，即压缩后的q和kv的隐向量维度，此时需要split。
    //                 https://zhuanlan.zhihu.com/p/25449691772
    // If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if (num_n_blocks <= 4) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    // <NT> 计算每个splits数所对应的使用率
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        float n_waves = float(total_mblocks * num_splits) / num_SMs;
        float eff = n_waves / ceil(n_waves);
        // printf("num_splits = %d, eff = %f\n", num_splits, eff);
        if (eff > max_efficiency) { max_efficiency = eff; }
        efficiency.push_back(eff);
    }
    // <NT> 选择满足85%利用率的最小拆分数
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}
