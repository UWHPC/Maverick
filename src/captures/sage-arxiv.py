"""Edit kernel bodies here. The runner reads this file without importing it."""

import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

WORKLOAD = {'fixture': 'arxiv-full',
 'architecture': 'sage',
 'dataset': 'ogbn-arxiv',
 'nodes': 169343,
 'edges': 2315598,
 'model': {'architecture': 'sage',
           'in_channels': 128,
           'out_channels': 40,
           'hidden_channels': 64,
           'dropout': 0.5,
           'attention_heads': 4,
           'attention_head_width': 8},
 'checkpoint_sha256': 'ab6cb665601fcd94b4cb43c43eabe914d0b00af902b64c1731ad6e27d55f62a3',
 'graph_sha256': '2afa99103d7100e5b1793b60fd18f4519216d3c5d8b2b20c197b6a153dee4b44',
 'versions': {'torch': '2.8.0+cu128',
              'pyg': '2.7.0',
              'triton': '3.4.0',
              'cuda': '12.8'},
 'checkpoint_dir': '../../runs/756588-TiByBQHQ/sage-arxiv',
 'data_dir': '../../ogb-cache'}

@triton.jit
def triton_poi_fused_index_select_new_zeros_scatter_add_0(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 21675904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_index_select_new_zeros_scatter_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 296396544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 128
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (2315598 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 169343")
    tmp3 = tl.full([XBLOCK], 169343, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 169343")
    tmp8 = tl.load(in_ptr1 + (tmp6 + 169343*x0), xmask, eviction_policy='evict_last')
    tl.atomic_add(out_ptr0 + (x0 + 128*tmp0), tmp8, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_new_ones_new_zeros_scatter_add_2(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 169343
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_new_ones_new_zeros_scatter_add_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2315598
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2315598 + x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 169343")
    tmp2 = 1.0
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp0, [XBLOCK])), tmp2, xmask, sem='relaxed')
    tl.atomic_add(out_ptr1 + (tl.broadcast_to(tmp0, [XBLOCK])), tmp2, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_div_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 21675904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = (tmp0 / tmp3)
    tl.store(out_ptr0 + (x2), tmp4, xmask)

@triton.jit
def triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10837952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)

@triton.jit
def triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 148198272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (2315598 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 169343")
    tmp3 = tl.full([XBLOCK], 169343, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 169343)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 169343")
    tmp8 = tl.load(in_ptr1 + (x0 + 64*tmp6), xmask)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.load(in_ptr3 + (x0 + 64*tmp6), xmask)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tl.atomic_add(out_ptr0 + (x0 + 64*tmp0), tmp14, xmask, sem='relaxed')

@triton.jit
def triton_poi_fused_div_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10837952
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
    xnumel = 10837952
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

_ORIGINALS = {'triton_poi_fused_index_select_new_zeros_scatter_add_0': ('4e5c493717ce38f400b72be62d3ba8efa7159cf775edb9eb15794b5094c55d7c',
                                                           "('out_ptr0, "
                                                           'xnumel, XBLOCK: '
                                                           "tl.constexpr', "
                                                           'None, None)'),
 'triton_poi_fused_index_select_new_zeros_scatter_add_1': ('00d60ee86cc4bfab50c13d7d3124d6dfddcfebf3d062c6c2416e1e427499ba5a',
                                                           "('in_ptr0, "
                                                           'in_ptr1, out_ptr0, '
                                                           'xnumel, XBLOCK: '
                                                           "tl.constexpr', "
                                                           'None, None)'),
 'triton_poi_fused_new_ones_new_zeros_scatter_add_2': ('b7cc076cf19eca0d96058a913a79026987a57ae5f38b38eb2445a92285b6459a',
                                                       "('out_ptr0, xnumel, "
                                                       "XBLOCK: tl.constexpr', "
                                                       'None, None)'),
 'triton_poi_fused_new_ones_new_zeros_scatter_add_3': ('ad9f38a43789795b50b81fdc08fe06c8190993312fa1993cb9e3db8299984072',
                                                       "('in_ptr0, out_ptr0, "
                                                       'out_ptr1, xnumel, '
                                                       "XBLOCK: tl.constexpr', "
                                                       'None, None)'),
 'triton_poi_fused_div_4': ('81c7b6fecbfb3b318e51aa974a6bdd8c995de4bb71591813fed0bf1d5a2df84d',
                            "('in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: "
                            "tl.constexpr', None, None)"),
 'triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5': ('947f3262ff02d5d727565868d784037a256e06ebb6c0fba95f7d545a20f64e3d',
                                                                          "('out_ptr0, "
                                                                          'xnumel, '
                                                                          'XBLOCK: '
                                                                          "tl.constexpr', "
                                                                          'None, '
                                                                          'None)'),
 'triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6': ('ca2fa1fb31d0e24cd048198273691a73a583efb62d12233ee3d8ba43accf968b',
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
 'triton_poi_fused_div_7': ('e2f871ea3ac21b194682d304d15b3803fff3c15e127e81fe0a4708545543b5af',
                            "('in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: "
                            "tl.constexpr', None, None)"),
 'triton_poi_fused_add_addmm_relu_8': ('ddcd196fe38518dc82e14fd3f797a31a5f383514d3284126f372fbfd368c4958',
                                       "('in_out_ptr0, in_ptr0, in_ptr1, "
                                       "xnumel, XBLOCK: tl.constexpr', None, "
                                       'None)')}

KERNELS = [triton_poi_fused_index_select_new_zeros_scatter_add_0, triton_poi_fused_index_select_new_zeros_scatter_add_1, triton_poi_fused_new_ones_new_zeros_scatter_add_2, triton_poi_fused_new_ones_new_zeros_scatter_add_3, triton_poi_fused_div_4, triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5, triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6, triton_poi_fused_div_7, triton_poi_fused_add_addmm_relu_8]
