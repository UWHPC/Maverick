"""Edit kernel bodies here. The runner reads this file without importing it."""

import triton
import triton.language as tl
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_helpers import libdevice

WORKLOAD = {'fixture': 'arxiv-full',
 'architecture': 'gat',
 'dataset': 'ogbn-arxiv',
 'nodes': 169343,
 'edges': 2484941,
 'model': {'architecture': 'gat',
           'in_channels': 128,
           'out_channels': 40,
           'hidden_channels': 64,
           'dropout': 0.5,
           'attention_heads': 4,
           'attention_head_width': 8},
 'checkpoint_sha256': 'baa8dfac3791172bbfb12155e0b067d83ec2e6228c94d55ab41e1e84b28294f7',
 'graph_sha256': '31bfd78f2d3d9acfeb540d4e3cc69a8846a408a026fcb357cefadaf0b9af0567',
 'versions': {'torch': '2.8.0+cu128',
              'pyg': '2.7.0',
              'triton': '3.4.0',
              'cuda': '12.8'},
 'checkpoint_dir': '../../runs/756588-TiByBQHQ/gat-arxiv',
 'data_dir': '../../ogb-cache',
 'capture_tolerance': {'atol': 1e-05, 'rtol': 0.0001}}

@triton.jit
def triton_per_fused_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 677372
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 8*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r0_2 + 8*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp12, xmask)

@triton.jit
def triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_1(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 677372
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_add_index_select_leaky_relu_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 9939764
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2484941 + x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 169343, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 169343")
    tmp6 = tl.load(in_ptr1 + (x0 + 4*tmp4), xmask)
    tmp8 = tmp7 + tmp1
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 169343")
    tmp12 = tl.load(in_ptr2 + (x0 + 4*tmp10), xmask)
    tmp13 = tmp6 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.2
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, xmask)

@triton.jit
def triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 9939764
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (2484941 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 169343")
    tmp3 = tl.full([XBLOCK], 169343, tl.int32)
    tmp4 = tmp0 + tmp3
    tmp5 = tmp0 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp0)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 169343")
    tmp8 = tl.load(in_ptr2 + (x0 + 4*tmp6), xmask)
    tmp9 = tmp2 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tl.atomic_add(out_ptr0 + (x0 + 4*tmp0), tmp10, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_4(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 5418976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 79518112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 32
    x4 = xindex // 8
    x1 = ((xindex // 8) % 4)
    x3 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (2484941 + x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 169343")
    tmp3 = tl.full([XBLOCK], 169343, tl.int32)
    tmp4 = tmp0 + tmp3
    tmp5 = tmp0 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp0)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 169343")
    tmp8 = tl.load(in_ptr2 + (x1 + 4*tmp6), xmask, eviction_policy='evict_last')
    tmp9 = tmp2 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.load(in_ptr3 + (x1 + 4*tmp6), xmask, eviction_policy='evict_last')
    tmp12 = 1e-16
    tmp13 = tmp11 + tmp12
    tmp14 = (tmp10 / tmp13)
    tmp16 = tmp15 + tmp3
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tl.device_assert(((0 <= tmp18) & (tmp18 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp18 < 169343")
    tmp20 = tl.load(in_ptr4 + (x3 + 32*tmp18), xmask)
    tmp21 = tmp14 * tmp20
    tl.atomic_add(out_ptr0 + (x3 + 32*tmp0), tmp21, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_add_elu_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 5418976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 1.0
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.expm1(tmp6)
    tmp8 = tmp7 * tmp5
    tmp9 = tl.where(tmp4, tmp6, tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)

@triton.jit
def triton_per_fused_mul_sum_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 169343
    r0_numel = 40
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 40*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)

@triton.jit
def triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_8(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 169343
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_add_index_select_leaky_relu_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2484941
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp7 = tl.load(in_ptr0 + (2484941 + x0), xmask)
    tmp1 = tl.full([XBLOCK], 169343, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 169343")
    tmp6 = tl.load(in_ptr1 + (tmp4), xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 169343")
    tmp12 = tl.load(in_ptr2 + (tmp10), xmask, eviction_policy='evict_last')
    tmp13 = tmp6 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.2
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(out_ptr0 + (x0), tmp18, xmask)

@triton.jit
def triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2484941
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2484941 + x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 169343")
    tmp3 = tl.full([XBLOCK], 169343, tl.int32)
    tmp4 = tmp0 + tmp3
    tmp5 = tmp0 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp0)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 169343")
    tmp8 = tl.load(in_ptr2 + (tmp6), xmask, eviction_policy='evict_last')
    tmp9 = tmp2 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp0, [XBLOCK])), tmp10, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_11(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6773720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 99397640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 40
    x0 = (xindex % 40)
    tmp0 = tl.load(in_ptr0 + (2484941 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 169343")
    tmp3 = tl.full([XBLOCK], 169343, tl.int32)
    tmp4 = tmp0 + tmp3
    tmp5 = tmp0 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp0)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 169343")
    tmp8 = tl.load(in_ptr2 + (tmp6), xmask, eviction_policy='evict_last')
    tmp9 = tmp2 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.load(in_ptr3 + (tmp6), xmask, eviction_policy='evict_last')
    tmp12 = 1e-16
    tmp13 = tmp11 + tmp12
    tmp14 = (tmp10 / tmp13)
    tmp16 = tmp15 + tmp3
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tl.device_assert(((0 <= tmp18) & (tmp18 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp18 < 169343")
    tmp20 = tl.load(in_ptr4 + (x0 + 40*tmp18), xmask)
    tmp21 = tmp14 * tmp20
    tl.atomic_add(out_ptr0 + (x0 + 40*tmp0), tmp21, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_add_mean_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6773720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = (tmp0 / tmp1)
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)

_ORIGINALS = {'triton_per_fused_mul_sum_0': ('a68efa0b6154f48737cd369b636d42d6d0d5bad3b5c295b99df1fe1766aec507',
                                "('in_ptr0, in_ptr1, in_ptr2, out_ptr0, "
                                'out_ptr1, xnumel, r0_numel, XBLOCK: '
                                "tl.constexpr', None, None)"),
 'triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_1': ('f67f23704b9648d62ed23604e2efa200ec35502b2e6a87fe227f86195bc8b59e',
                                                                             "('out_ptr0, "
                                                                             'xnumel, '
                                                                             'XBLOCK: '
                                                                             "tl.constexpr', "
                                                                             'None, '
                                                                             'None)'),
 'triton_poi_fused_add_index_select_leaky_relu_2': ('2c38bda2094ea724ebe9c980c933f61a5f8cd22f761d315e63e5fa9d8b93ead9',
                                                    "('in_ptr0, in_ptr1, "
                                                    'in_ptr2, out_ptr0, '
                                                    'xnumel, XBLOCK: '
                                                    "tl.constexpr', None, "
                                                    'None)'),
 'triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_3': ('c68b57993ba6e7daca5fd52c869bd1fccfb1a2b0917d9486fc7f9d331ceb99ae',
                                                                   "('in_ptr0, "
                                                                   'in_ptr1, '
                                                                   'in_ptr2, '
                                                                   'out_ptr0, '
                                                                   'xnumel, '
                                                                   'XBLOCK: '
                                                                   "tl.constexpr', "
                                                                   'None, '
                                                                   'None)'),
 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_4': ('7d7ba94ce4ca74676e1ea5c907efd4aad5266badd1d9879308ad324e794be5a2',
                                                               "('out_ptr0, "
                                                               'xnumel, '
                                                               'XBLOCK: '
                                                               "tl.constexpr', "
                                                               'None, None)'),
 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_5': ('9f1d82dbf9f3f95a98e39839c436cc23bf929cfb06fc70b21ea906e23fef0db3',
                                                               "('in_ptr0, "
                                                               'in_ptr1, '
                                                               'in_ptr2, '
                                                               'in_ptr3, '
                                                               'in_ptr4, '
                                                               'out_ptr0, '
                                                               'xnumel, '
                                                               'XBLOCK: '
                                                               "tl.constexpr', "
                                                               'None, None)'),
 'triton_poi_fused_add_elu_6': ('254b039d3d3a16eb9a6eb1d603185e7051d7e30282f31a3ad54be1fa282ba559',
                                "('in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: "
                                "tl.constexpr', None, None)"),
 'triton_per_fused_mul_sum_7': ('c801ed0159e2271c304c05dd07b196c0e0b2e2b87edbff0080156fe1aefe2f14',
                                "('in_ptr0, in_ptr1, in_ptr2, out_ptr0, "
                                'out_ptr1, xnumel, r0_numel, XBLOCK: '
                                "tl.constexpr', None, None)"),
 'triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_8': ('b7cc076cf19eca0d96058a913a79026987a57ae5f38b38eb2445a92285b6459a',
                                                                             "('out_ptr0, "
                                                                             'xnumel, '
                                                                             'XBLOCK: '
                                                                             "tl.constexpr', "
                                                                             'None, '
                                                                             'None)'),
 'triton_poi_fused_add_index_select_leaky_relu_9': ('94214b2046541ecaac898269c7e5f7f8f3b055c4ff8e6d6d896c4a28a56d646d',
                                                    "('in_ptr0, in_ptr1, "
                                                    'in_ptr2, out_ptr0, '
                                                    'xnumel, XBLOCK: '
                                                    "tl.constexpr', None, "
                                                    'None)'),
 'triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_10': ('7b6fea64731f76a81c81c1f641dccdf198a28cc72fa19a67be047d11aff407ff',
                                                                    "('in_ptr0, "
                                                                    'in_ptr1, '
                                                                    'in_ptr2, '
                                                                    'out_ptr0, '
                                                                    'xnumel, '
                                                                    'XBLOCK: '
                                                                    "tl.constexpr', "
                                                                    'None, '
                                                                    'None)'),
 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_11': ('65d27e4f91549963ff65e8bac47ad246c94ced7c9b7ba344d318ad7a2c5896b4',
                                                                "('out_ptr0, "
                                                                'xnumel, '
                                                                'XBLOCK: '
                                                                "tl.constexpr', "
                                                                'None, None)'),
 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_12': ('aca321c07d74b9b7b38e460a578bfc0c9a85b7db26780a5e191f700bc5311115',
                                                                "('in_ptr0, "
                                                                'in_ptr1, '
                                                                'in_ptr2, '
                                                                'in_ptr3, '
                                                                'in_ptr4, '
                                                                'out_ptr0, '
                                                                'xnumel, '
                                                                'XBLOCK: '
                                                                "tl.constexpr', "
                                                                'None, None)'),
 'triton_poi_fused_add_mean_13': ('390c9849bdf43f05b72cf7ba75e0eec95d9e2f6abedc284a144771f7d9ef762a',
                                  "('in_ptr0, in_ptr1, out_ptr0, xnumel, "
                                  "XBLOCK: tl.constexpr', None, None)")}

KERNELS = [triton_per_fused_mul_sum_0, triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_1, triton_poi_fused_add_index_select_leaky_relu_2, triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_3, triton_poi_fused_index_select_mul_new_zeros_scatter_add_4, triton_poi_fused_index_select_mul_new_zeros_scatter_add_5, triton_poi_fused_add_elu_6, triton_per_fused_mul_sum_7, triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_8, triton_poi_fused_add_index_select_leaky_relu_9, triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_10, triton_poi_fused_index_select_mul_new_zeros_scatter_add_11, triton_poi_fused_index_select_mul_new_zeros_scatter_add_12, triton_poi_fused_add_mean_13]
