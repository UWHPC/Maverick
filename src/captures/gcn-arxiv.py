"""Edit kernel bodies here. The runner reads this file without importing it."""

import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

WORKLOAD = {'fixture': 'arxiv-full',
 'architecture': 'gcn',
 'dataset': 'ogbn-arxiv',
 'nodes': 169343,
 'edges': 2484941,
 'model': {'architecture': 'gcn',
           'in_channels': 128,
           'out_channels': 40,
           'hidden_channels': 64,
           'dropout': 0.5,
           'attention_heads': 4,
           'attention_head_width': 8},
 'checkpoint_sha256': '3b32979282ec062abf0a5be1368e5853857717f1e9c0084d30f1c215fe95c433',
 'graph_sha256': '488b73e7fc983ff8e49876273b102e4fb2bbdbb44c99bcddd0d7e0758ea00503',
 'versions': {'torch': '2.8.0+cu128',
              'pyg': '2.7.0',
              'triton': '3.4.0',
              'cuda': '12.8'},
 'checkpoint_dir': '../../runs/756588-TiByBQHQ/gcn-arxiv',
 'data_dir': '../../ogb-cache',
 'capture_tolerance': {'atol': 1e-05, 'rtol': 0.0001}}

@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_0(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10837952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 159036224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (2484941 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 169343")
    tmp4 = tl.full([XBLOCK], 169343, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 169343")
    tmp9 = tl.load(in_ptr2 + (x0 + 64*tmp7), xmask)
    tmp10 = tmp2 * tmp9
    tl.atomic_add(out_ptr0 + (x0 + 64*tmp0), tmp10, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_add_relu_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10837952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2), tmp4, xmask)

@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_3(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6773720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 99397640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 40
    x0 = (xindex % 40)
    tmp0 = tl.load(in_ptr0 + (2484941 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 169343")
    tmp4 = tl.full([XBLOCK], 169343, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 169343")
    tmp9 = tl.load(in_ptr2 + (x0 + 40*tmp7), xmask)
    tmp10 = tmp2 * tmp9
    tl.atomic_add(out_ptr0 + (x0 + 40*tmp0), tmp10, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_add_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6773720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)

_ORIGINALS = {'triton_poi_fused_index_select_mul_new_zeros_scatter_add_0': ('947f3262ff02d5d727565868d784037a256e06ebb6c0fba95f7d545a20f64e3d',
                                                               "('out_ptr0, "
                                                               'xnumel, '
                                                               'XBLOCK: '
                                                               "tl.constexpr', "
                                                               'None, None)'),
 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_1': ('b0783298c76e0628a5a5aacc9d40b5b52dd043a4563ec6abf71a740987149769',
                                                               "('in_ptr0, "
                                                               'in_ptr1, '
                                                               'in_ptr2, '
                                                               'out_ptr0, '
                                                               'xnumel, '
                                                               'XBLOCK: '
                                                               "tl.constexpr', "
                                                               'None, None)'),
 'triton_poi_fused_add_relu_2': ('7cb8677c6670da0682dbe35e43c91d02d4f51bf5481085c7be7a23a5557ccf8c',
                                 "('in_ptr0, in_ptr1, out_ptr0, xnumel, "
                                 "XBLOCK: tl.constexpr', None, None)"),
 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_3': ('65d27e4f91549963ff65e8bac47ad246c94ced7c9b7ba344d318ad7a2c5896b4',
                                                               "('out_ptr0, "
                                                               'xnumel, '
                                                               'XBLOCK: '
                                                               "tl.constexpr', "
                                                               'None, None)'),
 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_4': ('f71aabcff4d1164f40bb434a641965065429b3204fb0579264175ec34f4b900d',
                                                               "('in_ptr0, "
                                                               'in_ptr1, '
                                                               'in_ptr2, '
                                                               'out_ptr0, '
                                                               'xnumel, '
                                                               'XBLOCK: '
                                                               "tl.constexpr', "
                                                               'None, None)'),
 'triton_poi_fused_add_5': ('42ea3941027acc6c557e7752dbdd63dec753ff7c996db5b849b2c2b49f07188b',
                            "('in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: "
                            "tl.constexpr', None, None)")}

KERNELS = [triton_poi_fused_index_select_mul_new_zeros_scatter_add_0, triton_poi_fused_index_select_mul_new_zeros_scatter_add_1, triton_poi_fused_add_relu_2, triton_poi_fused_index_select_mul_new_zeros_scatter_add_3, triton_poi_fused_index_select_mul_new_zeros_scatter_add_4, triton_poi_fused_add_5]
