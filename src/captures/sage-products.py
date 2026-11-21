"""Edit kernel bodies here. The runner reads this file without importing it."""

import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

WORKLOAD = {'fixture': 'products-250k',
 'architecture': 'sage',
 'dataset': 'ogbn-products',
 'nodes': 250000,
 'edges': 1289372,
 'model': {'architecture': 'sage',
           'in_channels': 100,
           'out_channels': 47,
           'hidden_channels': 64,
           'dropout': 0.5,
           'attention_heads': 4,
           'attention_head_width': 8},
 'checkpoint_sha256': '104ae7303d4decbbd00c316f54ae4fac9aa42f04f0e5323b2034f9f3aaa213a1',
 'graph_sha256': 'd1b698d82448439f962c5bed80bf3edb48b5e947e38c39db033bb7347a273891',
 'versions': {'torch': '2.8.0+cu128',
              'pyg': '2.7.0',
              'triton': '3.4.0',
              'cuda': '12.8'},
 'checkpoint_dir': '../../runs/756588-TiByBQHQ/sage-products',
 'data_dir': '../../ogb-cache',
 'capture_tolerance': {'atol': 0.0003, 'rtol': 0.0001}}

@triton.jit
def triton_poi_fused_index_select_new_zeros_scatter_add_0(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 25000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_index_select_new_zeros_scatter_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128937200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 100
    x0 = (xindex % 100)
    tmp0 = tl.load(in_ptr0 + (1289372 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 250000)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 250000")
    tmp3 = tl.full([XBLOCK], 250000, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 250000)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 250000")
    tmp8 = tl.load(in_ptr1 + (x0 + 100*tmp6), xmask)
    tl.atomic_add(out_ptr0 + (x0 + 100*tmp0), tmp8, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_new_ones_new_zeros_scatter_add_2(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 250000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_new_ones_new_zeros_scatter_add_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1289372
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1289372 + x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 250000)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 250000")
    tmp2 = 1.0
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp0, [XBLOCK])), tmp2, xmask, sem='relaxed')
    tl.atomic_add(out_ptr1 + (tl.broadcast_to(tmp0, [XBLOCK])), tmp2, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_div_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 25000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 100
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = (tmp0 / tmp3)
    tl.store(out_ptr0 + (x2), tmp4, xmask)

@triton.jit
def triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 82519808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (1289372 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 250000)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 250000")
    tmp3 = tl.full([XBLOCK], 250000, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 250000)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 250000")
    tmp8 = tl.load(in_ptr1 + (x0 + 64*tmp6), xmask)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.load(in_ptr3 + (x0 + 64*tmp6), xmask)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tl.atomic_add(out_ptr0 + (x0 + 64*tmp0), tmp14, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_div_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = (tmp0 / tmp3)
    tl.store(out_ptr0 + (x2), tmp4, xmask)

@triton.jit
def triton_poi_fused_add_addmm_relu_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)

_ORIGINALS = {'triton_poi_fused_index_select_new_zeros_scatter_add_0': ('44a23ac63f623811664b6b75e9cd8a18df9c8f995fdeb6e1c0724d10140d478d',
                                                           "('out_ptr0, "
                                                           'xnumel, XBLOCK: '
                                                           "tl.constexpr', "
                                                           'None, None)'),
 'triton_poi_fused_index_select_new_zeros_scatter_add_1': ('8585dd0617d12abd3ad0a29e00058f51d0f861276986ba0828e1c857488f8343',
                                                           "('in_ptr0, "
                                                           'in_ptr1, out_ptr0, '
                                                           'xnumel, XBLOCK: '
                                                           "tl.constexpr', "
                                                           'None, None)'),
 'triton_poi_fused_new_ones_new_zeros_scatter_add_2': ('1c25b72c94e36a6cf76fd51ac4f0c39ab32472fa96b0c47b1a8bb22a996bb769',
                                                       "('out_ptr0, xnumel, "
                                                       "XBLOCK: tl.constexpr', "
                                                       'None, None)'),
 'triton_poi_fused_new_ones_new_zeros_scatter_add_3': ('d9defd2baae1fc2e92b2093675b33c3f10bbc1d4eab5dbcb70fee5405a6eb8ed',
                                                       "('in_ptr0, out_ptr0, "
                                                       'out_ptr1, xnumel, '
                                                       "XBLOCK: tl.constexpr', "
                                                       'None, None)'),
 'triton_poi_fused_div_4': ('fefcef8b7337a04fc8df6d216559fecd5359a7793ba496ac3934ed7499b05aa7',
                            "('in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: "
                            "tl.constexpr', None, None)"),
 'triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5': ('6855260477ae386a44a1e59e9f476c4a7a5439ae8a5af0a9cbda00548eed09ba',
                                                                          "('out_ptr0, "
                                                                          'xnumel, '
                                                                          'XBLOCK: '
                                                                          "tl.constexpr', "
                                                                          'None, '
                                                                          'None)'),
 'triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6': ('a284774f2cf6deb22f60018f69ac4a57f4b190520df995acfaf252574416cc68',
                                                                          "('in_ptr0, "
                                                                          'in_ptr1, '
                                                                          'in_ptr2, '
                                                                          'in_ptr3, '
                                                                          'out_ptr0, '
                                                                          'xnumel, '
                                                                          'XBLOCK: '
                                                                          "tl.constexpr', "
                                                                          'None, '
                                                                          'None)'),
 'triton_poi_fused_div_7': ('9966ca66cdb2da0cf6d646d51e69ff23b8deebae5fb7970fd3822f9ed8332211',
                            "('in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: "
                            "tl.constexpr', None, None)"),
 'triton_poi_fused_add_addmm_relu_8': ('ea7a1eef33a1d3c2cce39857e9afe7ee7639298b343afbaa628f6fa050920ce7',
                                       "('in_out_ptr0, in_ptr0, in_ptr1, "
                                       "xnumel, XBLOCK: tl.constexpr', None, "
                                       'None)')}

KERNELS = [triton_poi_fused_index_select_new_zeros_scatter_add_0, triton_poi_fused_index_select_new_zeros_scatter_add_1, triton_poi_fused_new_ones_new_zeros_scatter_add_2, triton_poi_fused_new_ones_new_zeros_scatter_add_3, triton_poi_fused_div_4, triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5, triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6, triton_poi_fused_div_7, triton_poi_fused_add_addmm_relu_8]
