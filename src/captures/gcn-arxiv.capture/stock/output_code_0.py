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


# kernel path: cache/inductor/gf/cgfwutn2qipv7wlu2fvcglfqm67upuk4t75hgqkjhhdauc4j4vmw.py
# Topologically Sorted Source Nodes: [new_zeros, x_j, out, out_1], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros => full_default
#   out => mul
#   out_1 => scatter_add
#   x_j => index
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mm, [%select_1]), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %index), kwargs = {})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %expand, %mul), kwargs = {})
triton_poi_fused_index_select_mul_new_zeros_scatter_add_0 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 86703616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10837952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: cache/inductor/a7/ca7mje3gwfojt4c3wmyp3w6il34kch35dj2m46t5ydctgqbsljv3.py
# Topologically Sorted Source Nodes: [new_zeros, x_j, out, out_1], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros => full_default
#   out => mul
#   out_1 => scatter_add
#   x_j => index
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mm, [%select_1]), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %index), kwargs = {})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %expand, %mul), kwargs = {})
triton_poi_fused_index_select_mul_new_zeros_scatter_add_1 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/j6/cj6atx45k6oxf44x7vj2kkxanjb3yw4tl2ftjg46kbtebisi3i3w.py
# Topologically Sorted Source Nodes: [out_2, relu], Original ATen: [aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_2 => add
#   relu => relu
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%scatter_add, %arg4_1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
triton_poi_fused_add_relu_2 = async_compile.triton('triton_poi_fused_add_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_relu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 86703872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_relu_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/27/c277gnalffpnu4lf7p5ec2wej2hwg7rxjf74al7vkoyuccatyg6u.py
# Topologically Sorted Source Nodes: [new_zeros_1, x_j_1, out_3, out_4], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_1 => full_default_1
#   out_3 => mul_1
#   out_4 => scatter_add_1
#   x_j_1 => index_1
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 40], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mm_1, [%select_3]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %index_1), kwargs = {})
#   %scatter_add_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_1, 0, %expand_1, %mul_1), kwargs = {})
triton_poi_fused_index_select_mul_new_zeros_scatter_add_3 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 54189760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6773720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: cache/inductor/dc/cdcestqag22qzbtzzpzeggtiopw5jegifrnwwnczqicsoyj4x4tj.py
# Topologically Sorted Source Nodes: [new_zeros_1, x_j_1, out_3, out_4], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_1 => full_default_1
#   out_3 => mul_1
#   out_4 => scatter_add_1
#   x_j_1 => index_1
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 40], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mm_1, [%select_3]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %index_1), kwargs = {})
#   %scatter_add_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_1, 0, %expand_1, %mul_1), kwargs = {})
triton_poi_fused_index_select_mul_new_zeros_scatter_add_4 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_4', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/7l/c7lohpnep7axkzjyqmsmguhfwugdijsblnhdgqrrrlvdlsbu6lbt.py
# Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out_5 => add_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%scatter_add_1, %arg6_1), kwargs = {})
triton_poi_fused_add_5 = async_compile.triton('triton_poi_fused_add_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F9D44C12085C08D59390B12815E1013D8AA3EBA5E174AFD175306678DA24B0E1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 54189920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride(arg0_1, (169343, 128), (128, 1))
    assert_size_stride(arg1_1, (64, 128), (128, 1))
    assert_size_stride(arg2_1, (2, 2484941), (2484941, 1))
    assert_size_stride(arg3_1, (2484941, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (40, 64), (64, 1))
    assert_size_stride(arg6_1, (40, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((169343, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.mm]
        extern_kernels.mm(arg0_1, reinterpret_tensor(arg1_1, (128, 64), (1, 128), 0), out=buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((169343, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_zeros, x_j, out, out_1], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_0.run(buf1, 10837952, stream=stream0)
        # Topologically Sorted Source Nodes: [new_zeros, x_j, out, out_1], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_1.run(arg2_1, arg3_1, buf0, buf1, 159036224, stream=stream0)
        buf3 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [out_2, relu], Original ATen: [aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_relu_2.run(buf1, arg4_1, buf3, 10837952, stream=stream0)
        del arg4_1
        del buf1
        buf4 = empty_strided_cuda((169343, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2, relu, x_2], Original ATen: [aten.add, aten.relu, aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(arg5_1, (64, 40), (1, 64), 0), out=buf4)
        del arg5_1
        del buf3
        buf5 = empty_strided_cuda((169343, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_zeros_1, x_j_1, out_3, out_4], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_3.run(buf5, 6773720, stream=stream0)
        # Topologically Sorted Source Nodes: [new_zeros_1, x_j_1, out_3, out_4], Original ATen: [aten.new_zeros, aten.index_select, aten.mul, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_4.run(arg2_1, arg3_1, buf4, buf5, 99397640, stream=stream0)
        del arg2_1
        del arg3_1
        buf7 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_5.run(buf5, arg6_1, buf7, 6773720, stream=stream0)
        del arg6_1
        del buf5
    return (buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((169343, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, 2484941), (2484941, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((2484941, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((40, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
