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


# kernel path: cache/inductor/7f/c7fehvcfcya47erjnozepq6fk7oclk2nfjkadf3gpfjonqc6fhr3.py
# Topologically Sorted Source Nodes: [new_zeros_1, x_j, out], Original ATen: [aten.new_zeros, aten.index_select, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_1 => full_default_2
#   out => scatter_add_1
#   x_j => index
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%select_1]), kwargs = {})
#   %scatter_add_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_2, 0, %expand, %index), kwargs = {})
triton_poi_fused_index_select_new_zeros_scatter_add_0 = async_compile.triton('triton_poi_fused_index_select_new_zeros_scatter_add_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_new_zeros_scatter_add_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6AC1F511B5EF3A132B4136C5E897229092E0B0F9B95B85EBDE51C7804BB0E3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 173407232}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_new_zeros_scatter_add_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21675904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: cache/inductor/o7/co7bgvyuh2idmadytdspmggeinlyns6qz7pfmvatxlwsxgxdtldx.py
# Topologically Sorted Source Nodes: [new_zeros_1, x_j, out], Original ATen: [aten.new_zeros, aten.index_select, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_1 => full_default_2
#   out => scatter_add_1
#   x_j => index
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%select_1]), kwargs = {})
#   %scatter_add_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_2, 0, %expand, %index), kwargs = {})
triton_poi_fused_index_select_new_zeros_scatter_add_1 = async_compile.triton('triton_poi_fused_index_select_new_zeros_scatter_add_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_new_zeros_scatter_add_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '6AC1F511B5EF3A132B4136C5E897229092E0B0F9B95B85EBDE51C7804BB0E3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_new_zeros_scatter_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/g4/cg4h4aqavic2f6u7gj6ffrcqd6cfg4tfjxanumj5o2tv4cpauu7b.py
# Topologically Sorted Source Nodes: [count, new_ones, scatter_add_], Original ATen: [aten.new_zeros, aten.new_ones, aten.scatter_add]
# Source node to ATen node mapping:
#   count => full_default
#   new_ones => full_default_1
#   scatter_add_ => scatter_add
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2315598], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %select, %full_default_1), kwargs = {})
triton_poi_fused_new_ones_new_zeros_scatter_add_2 = async_compile.triton('triton_poi_fused_new_ones_new_zeros_scatter_add_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_ones_new_zeros_scatter_add_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6AC1F511B5EF3A132B4136C5E897229092E0B0F9B95B85EBDE51C7804BB0E3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1354744}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_ones_new_zeros_scatter_add_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 169343
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: cache/inductor/p6/cp6a3rffppspdqql42oblvrd3izkuv4ku5734uh3nrvixz5cljk7.py
# Topologically Sorted Source Nodes: [count, new_ones, scatter_add_, count_2, new_ones_1, scatter_add__2], Original ATen: [aten.new_zeros, aten.new_ones, aten.scatter_add]
# Source node to ATen node mapping:
#   count => full_default
#   count_2 => full_default_3
#   new_ones => full_default_1
#   new_ones_1 => full_default_4
#   scatter_add_ => scatter_add
#   scatter_add__2 => scatter_add_2
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2315598], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %select, %full_default_1), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2315598], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add_2 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_3, 0, %select_2, %full_default_4), kwargs = {})
triton_poi_fused_new_ones_new_zeros_scatter_add_3 = async_compile.triton('triton_poi_fused_new_ones_new_zeros_scatter_add_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_ones_new_zeros_scatter_add_3', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6AC1F511B5EF3A132B4136C5E897229092E0B0F9B95B85EBDE51C7804BB0E3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_ones_new_zeros_scatter_add_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/gj/cgjo344nmmaqvgnle5jnvjs67wdu2smmbhyhmuvkjyw7vxgxuuxu.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   out_1 => div
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%scatter_add_1, %expand_1), kwargs = {})
triton_poi_fused_div_4 = async_compile.triton('triton_poi_fused_div_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '6AC1F511B5EF3A132B4136C5E897229092E0B0F9B95B85EBDE51C7804BB0E3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 173407232}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/oq/coqoo3auegbfreor6xnr6nzy3neyqlpqic3l3cuhyydfubyxdxza.py
# Topologically Sorted Source Nodes: [new_zeros_3, out_2, out_3, relu, x_j_1, out_4], Original ATen: [aten.new_zeros, aten.addmm, aten.add, aten.relu, aten.index_select, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_3 => full_default_5
#   out_2 => add_tensor
#   out_3 => add
#   out_4 => scatter_add_3
#   relu => relu
#   x_j_1 => index_1
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg3_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor, %mm), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%relu, [%select_3]), kwargs = {})
#   %scatter_add_3 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_5, 0, %expand_2, %index_1), kwargs = {})
triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5 = async_compile.triton('triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6AC1F511B5EF3A132B4136C5E897229092E0B0F9B95B85EBDE51C7804BB0E3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 86703616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10837952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: cache/inductor/rr/crr23swynj3c4yi6c5z3hc4b55zsdzj6qwnlt7k5ppprg4bbqs5g.py
# Topologically Sorted Source Nodes: [new_zeros_3, out_2, out_3, relu, x_j_1, out_4], Original ATen: [aten.new_zeros, aten.addmm, aten.add, aten.relu, aten.index_select, aten.scatter_add]
# Source node to ATen node mapping:
#   new_zeros_3 => full_default_5
#   out_2 => add_tensor
#   out_3 => add
#   out_4 => scatter_add_3
#   relu => relu
#   x_j_1 => index_1
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([169343, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg3_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor, %mm), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%relu, [%select_3]), kwargs = {})
#   %scatter_add_3 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_5, 0, %expand_2, %index_1), kwargs = {})
triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6 = async_compile.triton('triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '6AC1F511B5EF3A132B4136C5E897229092E0B0F9B95B85EBDE51C7804BB0E3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/qb/cqbe4lu7swmnoyijaclhves3yvqwdqmipynp4worpro6gyjxw4ky.py
# Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   out_5 => div_1
# Graph fragment:
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%scatter_add_3, %expand_3), kwargs = {})
triton_poi_fused_div_7 = async_compile.triton('triton_poi_fused_div_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '6AC1F511B5EF3A132B4136C5E897229092E0B0F9B95B85EBDE51C7804BB0E3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 86703616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: cache/inductor/3n/c3nvha5pvdlzo2rtv6wlismgnhgekjlzq5hsekne2mtav2jn6ieo.py
# Topologically Sorted Source Nodes: [out_2, out_3, relu], Original ATen: [aten.addmm, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_2 => add_tensor
#   out_3 => add
#   relu => relu
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg3_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor, %mm), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
triton_poi_fused_add_addmm_relu_8 = async_compile.triton('triton_poi_fused_add_addmm_relu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '6AC1F511B5EF3A132B4136C5E897229092E0B0F9B95B85EBDE51C7804BB0E3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 173407488}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_relu_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    assert_size_stride(arg0_1, (169343, 128), (1, 169343))
    assert_size_stride(arg1_1, (2, 2315598), (2315598, 1))
    assert_size_stride(arg2_1, (64, 128), (128, 1))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, 128), (128, 1))
    assert_size_stride(arg5_1, (40, 64), (64, 1))
    assert_size_stride(arg6_1, (40, ), (1, ))
    assert_size_stride(arg7_1, (40, 64), (64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((169343, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_zeros_1, x_j, out], Original ATen: [aten.new_zeros, aten.index_select, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_new_zeros_scatter_add_0.run(buf0, 21675904, stream=stream0)
        # Topologically Sorted Source Nodes: [new_zeros_1, x_j, out], Original ATen: [aten.new_zeros, aten.index_select, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_new_zeros_scatter_add_1.run(arg1_1, arg0_1, buf0, 296396544, stream=stream0)
        buf2 = empty_strided_cuda((169343, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [count, new_ones, scatter_add_], Original ATen: [aten.new_zeros, aten.new_ones, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_ones_new_zeros_scatter_add_2.run(buf2, 169343, stream=stream0)
        buf9 = empty_strided_cuda((169343, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [count_2, new_ones_1, scatter_add__2], Original ATen: [aten.new_zeros, aten.new_ones, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_ones_new_zeros_scatter_add_2.run(buf9, 169343, stream=stream0)
        # Topologically Sorted Source Nodes: [count, new_ones, scatter_add_, count_2, new_ones_1, scatter_add__2], Original ATen: [aten.new_zeros, aten.new_ones, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_ones_new_zeros_scatter_add_3.run(arg1_1, buf2, buf9, 2315598, stream=stream0)
        buf4 = empty_strided_cuda((169343, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_4.run(buf0, buf2, buf4, 21675904, stream=stream0)
        del buf0
        del buf2
        buf5 = empty_strided_cuda((169343, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten.div, aten.addmm]
        extern_kernels.mm(buf4, reinterpret_tensor(arg2_1, (128, 64), (1, 128), 0), out=buf5)
        del arg2_1
        del buf4
        buf6 = empty_strided_cuda((169343, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(arg0_1, reinterpret_tensor(arg4_1, (128, 64), (1, 128), 0), out=buf6)
        del arg0_1
        del arg4_1
        buf7 = empty_strided_cuda((169343, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_zeros_3, out_2, out_3, relu, x_j_1, out_4], Original ATen: [aten.new_zeros, aten.addmm, aten.add, aten.relu, aten.index_select, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_5.run(buf7, 10837952, stream=stream0)
        # Topologically Sorted Source Nodes: [new_zeros_3, out_2, out_3, relu, x_j_1, out_4], Original ATen: [aten.new_zeros, aten.addmm, aten.add, aten.relu, aten.index_select, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_index_select_new_zeros_relu_scatter_add_6.run(arg1_1, buf5, arg3_1, buf6, buf7, 148198272, stream=stream0)
        del arg1_1
        buf11 = empty_strided_cuda((169343, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_7.run(buf7, buf9, buf11, 10837952, stream=stream0)
        del buf7
        del buf9
        buf12 = empty_strided_cuda((169343, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5, out_6], Original ATen: [aten.div, aten.addmm]
        extern_kernels.addmm(arg6_1, buf11, reinterpret_tensor(arg5_1, (64, 40), (1, 64), 0), alpha=1, beta=1, out=buf12)
        del arg5_1
        del arg6_1
        del buf11
        buf13 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [out_2, out_3, relu], Original ATen: [aten.addmm, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_relu_8.run(buf13, arg3_1, buf6, 10837952, stream=stream0)
        del arg3_1
        del buf6
        buf14 = empty_strided_cuda((169343, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2, out_3, relu], Original ATen: [aten.addmm, aten.add, aten.relu]
        extern_kernels.addmm(buf12, buf13, reinterpret_tensor(arg7_1, (64, 40), (1, 64), 0), alpha=1, beta=1, out=buf14)
        del arg7_1
        del buf12
        del buf13
    return (buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((169343, 128), (1, 169343), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 2315598), (2315598, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((40, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((40, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
