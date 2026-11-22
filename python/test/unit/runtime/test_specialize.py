import numpy
import pytest
import torch
from collections import namedtuple
from triton._C.libtriton import native_specialize_impl
from triton.runtime.jit import MockTensor, JITCallable
from triton._utils import canonicalize_dtype
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.amd.compiler import HIPBackend
from triton.language import constexpr
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor
from triton.experimental.gluon.language._layouts import NVMMASharedLayout


def mock_tensor_from_tensor(tensor):
    return MockTensor(tensor.dtype, tensor.shape)


class MockJITCallable(JITCallable):

    def __init__(self):
        pass

    def cache_key(self):
        return "mock_jit_callable"


class MockFloat(float):

    def __new__(cls, value):
        return super().__new__(cls, value)


class MockInt(int):

    def __new__(cls, value):
        return super().__new__(cls, value)


def reference_specialize_impl(backend, arg, is_const, specialize_value, align):
    if arg is None:
        return ("constexpr", None)
    elif isinstance(arg, bool):
        return ("u1", None)
    elif isinstance(arg, int):
        key = backend.get_int_specialization(arg, align=align) if specialize_value else None
        if arg == 1 and specialize_value:
            return ("constexpr", 1)
        elif -(2**31) <= arg and arg <= 2**31 - 1:
            return ("i32", key)
        elif 2**63 <= arg and arg <= 2**64 - 1:
            return ("u64", key)
        else:
            return ("i64", key)
    elif isinstance(arg, float):
        return ("fp32", None)
    elif hasattr(arg, "data_ptr"):
        dsk = (arg.dtype, is_const)
        res = ("*k" if dsk[1] else "*") + canonicalize_dtype(dsk[0])
        key = backend.get_tensor_specialization(arg, align=align) if specialize_value else None
        return (res, key)
    elif isinstance(arg, JITCallable):
        return ("constexpr", arg.cache_key)
    elif isinstance(arg, constexpr):
        return ("constexpr", arg)
    elif isinstance(arg, tuple):
        spec = [reference_specialize_impl(backend, x, False, True, True) for x in arg]
        make_tuple = lambda vals: type(arg)(*vals) if hasattr(arg, "_fields") else tuple(vals)
        tys = make_tuple([x[0] for x in spec])
        keys = make_tuple([x[1] for x in spec])
        return (tys, keys)
    elif isinstance(arg, TensorDescriptor):
        assert hasattr(arg.base, "data_ptr")
        inner = canonicalize_dtype(arg.base.dtype)
        return (f"tensordesc<{inner}{list(arg.block_shape)}>", None)
    elif isinstance(arg, GluonTensorDescriptor):
        assert hasattr(arg.base, "data_ptr")
        inner = canonicalize_dtype(arg.base.dtype)
        return (f"tensordesc<{inner}{list(arg.block_shape)},{arg.layout!r}>", None)
    else:
        raise TypeError("Unsupported type: %s" % type(arg))


def native_inputs_to_specialize():
    return [
        1.0,
        None,
        False,
        True,
        1,
        0,
        -1,
        16,
        17,
        2**31 - 1,
        2**31,
        -2 * 31 - 1,
        2**63 - 1,
        2**63,
        2**63 + 1,
        2**64 - 1,
    ]


def derived_inputs_to_specialize():
    return [
        constexpr(1),
        constexpr(False),
        constexpr(1.0),
        numpy.float64(1.0),
        MockFloat(1.0),
        MockInt(1),
        MockJITCallable(),
    ]


def tuples_to_specialize():
    return [
        (1, 1),
        (False, True),
        namedtuple('strides', ['x', 'y'])(1, 1),
        namedtuple('flags', ['x', 'y'])(False, True),
    ]


def tensors_to_specialize():
    return [
        torch.empty(shape, dtype=dtype, device="cpu")
        for shape in [(1, ), (1, 1), (16, ), (16, 16), (128, ), (128, 128)]
        for dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
    ]


def tensordescriptors_to_specialize():
    return [
        TensorDescriptor.from_tensor(tensor, block_shape=tensor.shape)
        for tensor in tensors_to_specialize()
        if tensor.shape[-1] % 16 == 0
    ]


def gluon_tensordescriptors_to_specialize():
    return [
        GluonTensorDescriptor.from_tensor(
            tensor,
            block_shape=tensor.shape,
            layout=NVMMASharedLayout(0, tensor.dtype.itemsize * 8, len(tensor.shape)),
        ) for tensor in tensors_to_specialize() if tensor.shape[-1] % 16 == 0 and tensor.dtype.itemsize <= 4
    ]


def mock_tensors_to_specialize():
    return [mock_tensor_from_tensor(tensor) for tensor in tensors_to_specialize()]


@pytest.mark.parametrize("input_generator", [
    native_inputs_to_specialize,
    tuples_to_specialize,
    tensors_to_specialize,
    tensordescriptors_to_specialize,
    gluon_tensordescriptors_to_specialize,
    mock_tensors_to_specialize,
])
@pytest.mark.parametrize("backend", [CUDABackend, HIPBackend])
@pytest.mark.parametrize("is_const", [True, False])
@pytest.mark.parametrize("specialize_value", [True, False])
@pytest.mark.parametrize("align", [True, False])
def test_specialize_impl(input_generator, backend, is_const, specialize_value, align):
    for arg in input_generator():
        result = native_specialize_impl(backend, arg, is_const, specialize_value, align)
        expected = reference_specialize_impl(backend, arg, is_const, specialize_value, align)
        assert result == expected


def test_annotated_constexpr_specialization_binder(monkeypatch):
    import triton
    import triton.language as tl
    import triton.runtime.jit as jit_mod

    calls = []
    orig_impl = jit_mod.native_specialize_impl

    def counting_impl(backend, arg, is_const, specialize_value, align):
        calls.append((arg, is_const, specialize_value, align))
        return orig_impl(backend, arg, is_const, specialize_value, align)

    monkeypatch.setattr(jit_mod, "native_specialize_impl", counting_impl)

    @triton.jit
    def k_ann(
        ptr: tl.pointer_type(tl.float32),
        opt_ptr: tl.pointer_type(tl.float32),
        stride: int,
        stride64: tl.int64,
        flag: bool,
        scale: float,
        C: tl.constexpr,
        _spec_0: int,
    ):
        pass

    @triton.jit(do_not_specialize=["stride", "stride64", "opt_ptr"])
    def k_dns(
        ptr: tl.pointer_type(tl.float32),
        opt_ptr: tl.pointer_type(tl.float32),
        stride: int,
        stride64: tl.int64,
    ):
        pass

    @triton.jit(do_not_specialize_on_alignment=["stride", "ptr"])
    def k_dna(
        ptr: tl.pointer_type(tl.float32),
        stride: int,
    ):
        pass

    backend = CUDABackend
    b_ann = jit_mod.create_function_from_signature(k_ann.signature, k_ann.params, backend)
    b_dns = jit_mod.create_function_from_signature(k_dns.signature, k_dns.params, backend)
    b_dna = jit_mod.create_function_from_signature(k_dna.signature, k_dna.params, backend)

    t = MockTensor(torch.float32, (16, ))

    calls.clear()
    _, spec, _ = b_ann(t, None, 1, 1, True, 0.5, 64, 1)
    assert spec == [
        ("*fp32", "D"),
        ("*fp32", None),
        ("constexpr", 1),
        ("constexpr", 1),
        ("u1", None),
        ("fp32", None),
        ("constexpr", 64),
        ("constexpr", 1),
    ]
    # 5 runtime-annotated params (ptr, opt_ptr, stride, stride64, _spec_0);
    # flag (bool), scale (float), and C (tl.constexpr) skip native_specialize_impl.
    assert len(calls) == 5

    calls.clear()
    _, spec, _ = b_ann(t, t, 2, 16, False, 1.0, 64, 7)
    assert spec == [
        ("*fp32", "D"),
        ("*fp32", "D"),
        ("i32", ""),
        ("i64", "D"),
        ("u1", None),
        ("fp32", None),
        ("constexpr", 64),
        ("i32", ""),
    ]
    assert len(calls) == 5

    calls.clear()
    _, spec, _ = b_dns(t, None, 1, 1)
    assert spec == [("*fp32", "D"), ("*fp32", None), ("i32", None), ("i64", None)]
    assert [specialize_value for _, _, specialize_value, _ in calls] == [True, False, False, False]

    calls.clear()
    _, spec, _ = b_dns(t, t, 16, 16)
    assert spec == [("*fp32", "D"), ("*fp32", None), ("i32", None), ("i64", None)]
    assert len(calls) == 4

    calls.clear()
    _, spec, _ = b_dna(t, 1)
    assert spec == [("*fp32", ""), ("constexpr", 1)]
    assert len(calls) == 2

    calls.clear()
    _, spec, _ = b_dna(t, 16)
    assert spec == [("*fp32", ""), ("i32", "")]
    assert len(calls) == 2


def test_annotated_constexpr_specialization_ttir_and_run(device):
    import triton
    import triton.language as tl

    def _tt_func_header(compiled):
        for line in compiled.asm["ttir"].splitlines():
            if "tt.func" in line:
                return line.strip()
        raise AssertionError("missing tt.func in TTIR")

    @triton.jit
    def k_stride(
        out_ptr: tl.pointer_type(tl.float32),
        in_ptr: tl.pointer_type(tl.float32),
        opt_bias: tl.pointer_type(tl.float32),
        stride: int,
        stride64: tl.int64,
        flag: tl.int1,
        scale: float,
        BLOCK: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK)
        val = tl.load(in_ptr + offs * stride) * scale
        if opt_bias.to(tl.uint64) != 0:
            val += tl.load(opt_bias + offs * stride64)
        if flag:
            val += 1.0
        tl.store(out_ptr + offs, val)

    @triton.jit(do_not_specialize=["stride"])
    def k_stride_dns(out_ptr: tl.pointer_type(tl.float32), in_ptr: tl.pointer_type(tl.float32), stride: int,
                     BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(in_ptr + offs * stride))

    inp = torch.arange(64, dtype=torch.float32, device=device)
    bias = torch.full((64, ), 10.0, dtype=torch.float32, device=device)
    out = torch.empty(4, dtype=torch.float32, device=device)

    # Integer unit strides are folded while opt_bias remains a nullable pointer.
    h1 = k_stride[(1, )](out, inp, None, 1, 1, True, 2.0, 4)
    sig1 = _tt_func_header(h1)
    assert "%stride:" not in sig1
    assert "%stride64:" not in sig1
    assert "%opt_bias: !tt.ptr<f32>" in sig1
    assert "%flag: i1" in sig1
    assert "%scale: f32" in sig1
    torch.testing.assert_close(out, inp[:4] * 2.0 + 1.0)

    # stride=2 and stride64=16 stay runtime args with their annotated types and divisibility.
    h2 = k_stride[(1, )](out, inp, bias, 2, 16, False, 0.5, 4)
    sig2 = _tt_func_header(h2)
    assert "%opt_bias: !tt.ptr<f32>" in sig2
    assert "%stride: i32" in sig2
    assert "%stride64: i64 {tt.divisibility = 16 : i32}" in sig2
    torch.testing.assert_close(out, inp[0:8:2] * 0.5 + 10.0)

    # stride=16 gets i32 + divisibility=16.
    h16 = k_stride[(1, )](out, inp, None, 16, 2, False, 1.0, 4)
    sig16 = _tt_func_header(h16)
    assert "%stride: i32 {tt.divisibility = 16 : i32}" in sig16
    assert "%stride64: i64" in sig16
    assert "%opt_bias: !tt.ptr<f32>" in sig16
    torch.testing.assert_close(out, inp[0:64:16])

    # do_not_specialize keeps stride=1 as runtime i32.
    hdns = k_stride_dns[(1, )](out, inp, 1, 4)
    sig_dns = _tt_func_header(hdns)
    assert "%stride: i32" in sig_dns
    torch.testing.assert_close(out, inp[:4])


@pytest.mark.parametrize("input_dtype,output_dtype,maximum", [
    (torch.int8, torch.int32, 127),
    (torch.int32, torch.int64, 2**31 - 1),
])
def test_annotated_integer_specialization_preserves_promotion(device, input_dtype, output_dtype, maximum):
    import triton
    import triton.language as tl
    from triton.compiler import ASTSource

    @triton.jit
    def add_i32(out, inp, amount: int):
        tl.store(out, tl.load(inp) + amount)

    @triton.jit
    def add_i64(out, inp, amount: tl.int64):
        tl.store(out, tl.load(inp) + amount)

    kernel = add_i32 if output_dtype == torch.int32 else add_i64
    inp = torch.full((1, ), maximum, dtype=input_dtype, device=device)
    out = torch.empty((1, ), dtype=output_dtype, device=device)
    compiled = kernel[(1, )](out, inp, 1)

    assert out.item() == maximum + 1
    assert "%amount:" not in compiled.asm["ttir"]

    # An explicitly supplied ASTSource constant still has Python scalar semantics.
    source = compiled.src
    manual_source = ASTSource(kernel, source.signature, source.constants, source.attrs)
    assert source.hash() != manual_source.hash()


def test_annotated_nullable_pointer_remains_typed(device):
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(out, opt_ptr: tl.pointer_type(tl.float32)):
        tl.store(out, tl.load(opt_ptr, mask=False, other=3.0))

    out = torch.empty((1, ), dtype=torch.float32, device=device)
    compiled = kernel[(1, )](out, None)
    assert out.item() == 3.0
    assert "%opt_ptr: !tt.ptr<f32>" in compiled.asm["ttir"]
