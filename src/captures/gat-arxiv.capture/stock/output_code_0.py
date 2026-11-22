# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: cache/inductor/qp/cqpz5rpgqzv3wgm4m6nonroauabqdypmqjmm4tmlf23lfmfgztah.py
# Topologically Sorted Source Nodes: [mul, alpha_src, mul_1, alpha_dst], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   alpha_dst => sum_2
#   alpha_src => sum_1
#   mul => mul
#   mul_1 => mul_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %arg2_1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [-1]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %arg3_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [-1]), kwargs = {})
triton_per_fused_mul_sum_0 = async_compile.triton('triton_per_fused_mul_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10837952, 'r0_': 21676160}}
)
@triton.jit
def triton_per_fused_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/qj/cqjslxjwzaf5g5mot6vzhqdxtslkzay4yqrbmcupywl32eh2xwcb.py
# Topologically Sorted Source Nodes: [new_zeros, alpha_j, alpha_i, alpha, alpha_1, src_max], Original ATen: [aten.new_zeros, aten.index_select, aten.add, aten.leaky_relu, aten.scatter_reduce]
# Source node to ATen node mapping:
#   alpha => add
#   alpha_1 => gt, mul_2, where
#   alpha_i => index_1
#   alpha_j => index
#   new_zeros => full_default
#   src_max => scatter_reduce
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%sum_1, [%select_1]), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%sum_2, [%select]), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%index, %index_1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add, %mul_2), kwargs = {})
#   %scatter_reduce : [num_users=1] = call_function[target=torch.ops.aten.scatter_reduce.two](args = (%full_default, 0, %expand, %where, amax), kwargs = {include_self: False})
triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_1 = async_compile.triton('triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5418976}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 677372
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: cache/inductor/zv/czvn3s2r4yejk6xt43zbkzyat4ryhxe3yy7y3jkcnbolt7xxnegm.py
# Topologically Sorted Source Nodes: [alpha_j, alpha_i, alpha, alpha_1], Original ATen: [aten.index_select, aten.add, aten.leaky_relu]
# Source node to ATen node mapping:
#   alpha => add
#   alpha_1 => gt, mul_2, where
#   alpha_i => index_1
#   alpha_j => index
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%sum_1, [%select_1]), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%sum_2, [%select]), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%index, %index_1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add, %mul_2), kwargs = {})
triton_poi_fused_add_index_select_leaky_relu_2 = async_compile.triton('triton_poi_fused_add_index_select_leaky_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_leaky_relu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_select_leaky_relu_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/4g/c4geaxqxsdvrsmhjbcvmsxnbg4vapr422nmzbfvzp5h6jqsltbwa.py
# Topologically Sorted Source Nodes: [new_zeros_1, index_select_2, out, out_1, scatter_add_], Original ATen: [aten.new_zeros, aten.index_select, aten.sub, aten.exp, aten.scatter_add]
# Source node to ATen node mapping:
#   index_select_2 => index_2
#   new_zeros_1 => full_default_1
#   out => sub
#   out_1 => exp
#   scatter_add_ => scatter_add
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%scatter_reduce, [%select]), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %index_2), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_1, 0, %expand_1, %exp), kwargs = {})
triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_3 = async_compile.triton('triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_3', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/jf/cjfz44yfmlyredvg4pnq72qgsurrc6gapxrxbntots56vzdbmp4y.py
# Topologically Sorted Source Nodes: [new_zeros_2, x_j, out_2, out_3], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_2 => full_default_2
#   out_2 => mul_3
#   out_3 => scatter_add_1
#   x_j => index_4
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 4, 8], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_4 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view, [%select_3]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %index_4), kwargs = {})
#   %scatter_add_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_2, 0, %expand_2, %mul_3), kwargs = {})
triton_poi_fused_index_select_mul_new_zeros_scatter_add_4 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 43351808}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5418976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: cache/inductor/iq/ciqykww7bmsj2lvsbj23hipr33tgorez4slhbqlpgzkkjyvxesac.py
# Topologically Sorted Source Nodes: [new_zeros_2, x_j, out_2, out_3], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_2 => full_default_2
#   out_2 => mul_3
#   out_3 => scatter_add_1
#   x_j => index_4
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 4, 8], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_4 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view, [%select_3]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %index_4), kwargs = {})
#   %scatter_add_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_2, 0, %expand_2, %mul_3), kwargs = {})
triton_poi_fused_index_select_mul_new_zeros_scatter_add_5 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_5', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/kl/cklajgwj5225cv7unldo6q33ixi6s7i3t3q76kxpmkh3syaffswy.py
# Topologically Sorted Source Nodes: [out_5, elu], Original ATen: [aten.add, aten.elu]
# Source node to ATen node mapping:
#   elu => expm1, gt_1, mul_4, mul_5, mul_6, where_1
#   out_5 => add_2
# Graph fragment:
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %arg5_1), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_2, 0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 1.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_5,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %mul_4, %mul_6), kwargs = {})
triton_poi_fused_add_elu_6 = async_compile.triton('triton_poi_fused_add_elu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_elu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 43351936}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_elu_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/3b/c3b4fiqczmnhtmjqnyftrx6zxs7bwcdzbdn5dorwf5lyi3upisml.py
# Topologically Sorted Source Nodes: [mul_3, alpha_src_1, mul_4, alpha_dst_1], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   alpha_dst_1 => sum_4
#   alpha_src_1 => sum_3
#   mul_3 => mul_7
#   mul_4 => mul_8
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %arg7_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_7, [-1]), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %arg8_1), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_8, [-1]), kwargs = {})
triton_per_fused_mul_sum_7 = async_compile.triton('triton_per_fused_mul_sum_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2709488, 'r0_': 27095200}}
)
@triton.jit
def triton_per_fused_mul_sum_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/zu/czuyfgoq232rrjuqozqqzzqlwc6ccdlngfaxslj72onkxvsgvsa6.py
# Topologically Sorted Source Nodes: [new_zeros_3, alpha_j_1, alpha_i_1, alpha_4, alpha_5, src_max_1], Original ATen: [aten.new_zeros, aten.index_select, aten.add, aten.leaky_relu, aten.scatter_reduce]
# Source node to ATen node mapping:
#   alpha_4 => add_3
#   alpha_5 => gt_2, mul_9, where_2
#   alpha_i_1 => index_6
#   alpha_j_1 => index_5
#   new_zeros_3 => full_default_3
#   src_max_1 => scatter_reduce_1
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_5 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%sum_3, [%select_5]), kwargs = {})
#   %index_6 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%sum_4, [%select_4]), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%index_5, %index_6), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.2), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_3, %mul_9), kwargs = {})
#   %scatter_reduce_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter_reduce.two](args = (%full_default_3, 0, %expand_3, %where_2, amax), kwargs = {include_self: False})
triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_8 = async_compile.triton('triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1354744}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 169343
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: cache/inductor/5o/c5ox4gknfptv7xp42kdsqve4kevpq7qyo62vx4dwybrtcarilgod.py
# Topologically Sorted Source Nodes: [alpha_j_1, alpha_i_1, alpha_4, alpha_5], Original ATen: [aten.index_select, aten.add, aten.leaky_relu]
# Source node to ATen node mapping:
#   alpha_4 => add_3
#   alpha_5 => gt_2, mul_9, where_2
#   alpha_i_1 => index_6
#   alpha_j_1 => index_5
# Graph fragment:
#   %index_5 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%sum_3, [%select_5]), kwargs = {})
#   %index_6 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%sum_4, [%select_4]), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%index_5, %index_6), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.2), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_3, %mul_9), kwargs = {})
triton_poi_fused_add_index_select_leaky_relu_9 = async_compile.triton('triton_poi_fused_add_index_select_leaky_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_leaky_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_select_leaky_relu_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/q7/cq7ny6bacokotjo4sdoayr4rhzva6lrx6ziwayq72b6bp6enyddf.py
# Topologically Sorted Source Nodes: [new_zeros_4, index_select_7, out_6, out_7, scatter_add__2], Original ATen: [aten.new_zeros, aten.index_select, aten.sub, aten.exp, aten.scatter_add]
# Source node to ATen node mapping:
#   index_select_7 => index_7
#   new_zeros_4 => full_default_4
#   out_6 => sub_1
#   out_7 => exp_1
#   scatter_add__2 => scatter_add_2
# Graph fragment:
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_7 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%scatter_reduce_1, [%select_4]), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_2, %index_7), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %scatter_add_2 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_4, 0, %expand_4, %exp_1), kwargs = {})
triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_10 = async_compile.triton('triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_10', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/wa/cwagxgdzj7kalocuenkcte5oydkfkiu6fgbtt555xd2hicrva27f.py
# Topologically Sorted Source Nodes: [new_zeros_5, x_j_1, out_8, out_9], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_5 => full_default_5
#   out_8 => mul_10
#   out_9 => scatter_add_3
#   x_j_1 => index_9
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 1, 40], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_9 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_6, [%select_7]), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %index_9), kwargs = {})
#   %scatter_add_3 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_5, 0, %expand_5, %mul_10), kwargs = {})
triton_poi_fused_index_select_mul_new_zeros_scatter_add_11 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 54189760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_11(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6773720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: cache/inductor/kb/ckbv3vu2zcygvfs5v7vp2wkra37gviasi366nj7c4kifqfkuthue.py
# Topologically Sorted Source Nodes: [new_zeros_5, x_j_1, out_8, out_9], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_5 => full_default_5
#   out_8 => mul_10
#   out_9 => scatter_add_3
#   x_j_1 => index_9
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 1, 40], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_9 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_6, [%select_7]), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %index_9), kwargs = {})
#   %scatter_add_3 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_5, 0, %expand_5, %mul_10), kwargs = {})
triton_poi_fused_index_select_mul_new_zeros_scatter_add_12 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_12', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/u4/cu4ognn65oj4bhmjomswepfaw24gcu6y73666imkvpxm3kypkfeq.py
# Topologically Sorted Source Nodes: [out_10, out_11], Original ATen: [aten.mean, aten.add]
# Source node to ATen node mapping:
#   out_10 => mean
#   out_11 => add_5
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%scatter_add_3, [1]), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %arg9_1), kwargs = {})
triton_poi_fused_add_mean_13 = async_compile.triton('triton_poi_fused_add_mean_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 54189920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
    args.clear()
    assert_size_stride(arg0_1, (169343, 128), (128, 1))
    assert_size_stride(arg1_1, (32, 128), (128, 1))
    assert_size_stride(arg2_1, (1, 4, 8), (32, 8, 1))
    assert_size_stride(arg3_1, (1, 4, 8), (32, 8, 1))
    assert_size_stride(arg4_1, (2, 2484941), (2484941, 1))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (40, 32), (32, 1))
    assert_size_stride(arg7_1, (1, 1, 40), (40, 40, 1))
    assert_size_stride(arg8_1, (1, 1, 40), (40, 40, 1))
    assert_size_stride(arg9_1, (40, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((169343, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(arg0_1, reinterpret_tensor(arg1_1, (128, 32), (1, 128), 0), out=buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((169343, 4), (4, 1), torch.float32)
        buf2 = empty_strided_cuda((169343, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, alpha_src, mul_1, alpha_dst], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_0.run(buf0, arg2_1, arg3_1, buf1, buf2, 677372, 8, stream=stream0)
        del arg2_1
        del arg3_1
        buf3 = empty_strided_cuda((169343, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_zeros, alpha_j, alpha_i, alpha, alpha_1, src_max], Original ATen: [aten.new_zeros, aten.index_select, aten.add, aten.leaky_relu, aten.scatter_reduce]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_1.run(buf3, 677372, stream=stream0)
        buf4 = empty_strided_cuda((2484941, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_j, alpha_i, alpha, alpha_1], Original ATen: [aten.index_select, aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_index_select_leaky_relu_2.run(arg4_1, buf1, buf2, buf4, 9939764, stream=stream0)
        del buf1
        aten.scatter_reduce_.two(buf3,0,reinterpret_tensor(arg4_1, (2484941, 4), (1, 0), 2484941),buf4, reduce='amax', include_self=False)
        buf6 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [new_zeros_1, index_select_2, out, out_1, scatter_add_], Original ATen: [aten.new_zeros, aten.index_select, aten.sub, aten.exp, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_1.run(buf6, 677372, stream=stream0)
        # Topologically Sorted Source Nodes: [new_zeros_1, index_select_2, out, out_1, scatter_add_], Original ATen: [aten.new_zeros, aten.index_select, aten.sub, aten.exp, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_3.run(arg4_1, buf4, buf3, buf6, 9939764, stream=stream0)
        buf8 = empty_strided_cuda((169343, 4, 8), (32, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_zeros_2, x_j, out_2, out_3], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_4.run(buf8, 5418976, stream=stream0)
        # Topologically Sorted Source Nodes: [new_zeros_2, x_j, out_2, out_3], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_5.run(arg4_1, buf4, buf3, buf6, buf0, buf8, 79518112, stream=stream0)
        del buf3
        del buf4
        del buf6
        buf10 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [out_5, elu], Original ATen: [aten.add, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_elu_6.run(buf8, arg5_1, buf10, 5418976, stream=stream0)
        del arg5_1
        del buf8
        buf11 = empty_strided_cuda((169343, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5, elu, linear_1], Original ATen: [aten.add, aten.elu, aten.mm]
        extern_kernels.mm(buf10, reinterpret_tensor(arg6_1, (32, 40), (1, 32), 0), out=buf11)
        del arg6_1
        del buf10
        buf12 = empty_strided_cuda((169343, 1), (1, 169344), torch.float32)
        buf13 = empty_strided_cuda((169343, 1), (1, 169344), torch.float32)
        # Topologically Sorted Source Nodes: [mul_3, alpha_src_1, mul_4, alpha_dst_1], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_7.run(buf11, arg7_1, arg8_1, buf12, buf13, 169343, 40, stream=stream0)
        del arg7_1
        del arg8_1
        buf14 = empty_strided_cuda((169343, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_zeros_3, alpha_j_1, alpha_i_1, alpha_4, alpha_5, src_max_1], Original ATen: [aten.new_zeros, aten.index_select, aten.add, aten.leaky_relu, aten.scatter_reduce]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_8.run(buf14, 169343, stream=stream0)
        buf15 = empty_strided_cuda((2484941, 1), (1, 2484960), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_j_1, alpha_i_1, alpha_4, alpha_5], Original ATen: [aten.index_select, aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_index_select_leaky_relu_9.run(arg4_1, buf12, buf13, buf15, 2484941, stream=stream0)
        del buf12
        aten.scatter_reduce_.two(buf14,0,reinterpret_tensor(arg4_1, (2484941, 1), (1, 1), 2484941),buf15, reduce='amax', include_self=False)
        buf17 = reinterpret_tensor(buf13, (169343, 1), (1, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [new_zeros_4, index_select_7, out_6, out_7, scatter_add__2], Original ATen: [aten.new_zeros, aten.index_select, aten.sub, aten.exp, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_index_select_leaky_relu_new_zeros_scatter_reduce_8.run(buf17, 169343, stream=stream0)
        # Topologically Sorted Source Nodes: [new_zeros_4, index_select_7, out_6, out_7, scatter_add__2], Original ATen: [aten.new_zeros, aten.index_select, aten.sub, aten.exp, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_index_select_new_zeros_scatter_add_sub_10.run(arg4_1, buf15, buf14, buf17, 2484941, stream=stream0)
        buf19 = empty_strided_cuda((169343, 1, 40), (40, 40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_zeros_5, x_j_1, out_8, out_9], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_11.run(buf19, 6773720, stream=stream0)
        # Topologically Sorted Source Nodes: [new_zeros_5, x_j_1, out_8, out_9], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_12.run(arg4_1, buf15, buf14, buf17, buf11, buf19, 99397640, stream=stream0)
        del arg4_1
        del buf14
        del buf15
        del buf17
        buf21 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [out_10, out_11], Original ATen: [aten.mean, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_13.run(buf19, arg9_1, buf21, 6773720, stream=stream0)
        del arg9_1
        del buf19
    return (buf21, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((169343, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 4, 8), (32, 8, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 4, 8), (32, 8, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 2484941), (2484941, 1), device='cuda:0', dtype=torch.int64)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((40, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, 1, 40), (40, 40, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1, 1, 40), (40, 40, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
