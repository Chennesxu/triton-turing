// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s --check-prefix=RLC
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -allocate-shared-memory | FileCheck %s --check-prefix=ALLOC

// Positive 1: FlagTree #1333 loop_histogram on SM75.
// The loop-carried histogram accumulator moves onto the blocked parent
// synthesized from the coalesced 1D load ([1, 8]), eliminating the per-iter
// shared-memory round trip (scratch drops from 2048 B to 512 B on this branch).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // RLC-DAG: #[[$P:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
  // RLC-LABEL: tt.func public @loop_histogram
  // RLC: scf.for {{.*}} -> (tensor<16x1024xi64, #[[$P]]>)
  // RLC: ttg.convert_layout %{{.*}} : tensor<1024xi16, #blocked1> -> tensor<1024xi16, #ttg.slice<{dim = 0, parent = #[[$P]]}>>
  // RLC: scf.yield {{.*}} : tensor<16x1024xi64, #[[$P]]>
  // RLC: tt.reduce
  // RLC: ttg.convert_layout %{{.*}} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #[[$P]]}>> -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>

  // ALLOC: module attributes {{{.*}}ttg.shared = 512 : i32
  // ALLOC-LABEL: tt.func public @loop_histogram
  // ALLOC: scf.for
  // ALLOC-NOT: allocation.offset
  // ALLOC: scf.yield
  // ALLOC: tt.reduce
  tt.func public @loop_histogram(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %n: i32) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %zero = arith.constant dense<0> : tensor<16x1024xi64, #blocked>
    %fifteen = arith.constant dense<15> : tensor<1024xi16, #blocked1>
    %bins = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bins2 = tt.expand_dims %bins {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %binsb = tt.broadcast %bins2 : tensor<16x1xi32, #blocked> -> tensor<16x1024xi32, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %base = tt.splat %ptr : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked1>
    %acc = scf.for %i = %c0_i32 to %n step %c1024_i32 iter_args(%acc_i = %zero) -> (tensor<16x1024xi64, #blocked>) : i32 {
      %is = tt.splat %i : i32 -> tensor<1024xi32, #blocked1>
      %o = arith.addi %offs, %is : tensor<1024xi32, #blocked1>
      %p = tt.addptr %base, %o : tensor<1024x!tt.ptr<i16>, #blocked1>, tensor<1024xi32, #blocked1>
      %x = tt.load %p : tensor<1024x!tt.ptr<i16>, #blocked1>
      %key = arith.andi %x, %fifteen : tensor<1024xi16, #blocked1>
      %keyc = ttg.convert_layout %key : tensor<1024xi16, #blocked1> -> tensor<1024xi16, #ttg.slice<{dim = 0, parent = #blocked}>>
      %keyi = arith.extui %keyc : tensor<1024xi16, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %key2 = tt.expand_dims %keyi {axis = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xi32, #blocked>
      %keyb = tt.broadcast %key2 : tensor<1x1024xi32, #blocked> -> tensor<16x1024xi32, #blocked>
      %hit = arith.cmpi eq, %binsb, %keyb : tensor<16x1024xi32, #blocked>
      %hit64 = arith.extui %hit : tensor<16x1024xi1, #blocked> to tensor<16x1024xi64, #blocked>
      %next = arith.addi %acc_i, %hit64 : tensor<16x1024xi64, #blocked>
      scf.yield %next : tensor<16x1024xi64, #blocked>
    }
    %count = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
    ^bb0(%a: i64, %b: i64):
      %s = arith.addi %a, %b : i64
      tt.reduce.return %s : i64
    }) : (tensor<16x1024xi64, #blocked>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %count : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Positive 2: Non-[1,8], axis=1 expand_dims parent synthesis ([4] -> [4, 1]).
// Confirms parent layout is derived from the input encoding rather than
// hardcoded to [1, 8] or axis=0.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // RLC-DAG: #[[$P4:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
  // RLC-LABEL: tt.func public @loop_expand_axis1_spt4
  // RLC: scf.for {{.*}} -> (tensor<512x16xf32, #[[$P4]]>)
  // RLC: scf.yield {{.*}} : tensor<512x16xf32, #[[$P4]]>

  // ALLOC-LABEL: tt.func public @loop_expand_axis1_spt4
  // ALLOC: scf.for
  // ALLOC-NOT: allocation.offset
  // ALLOC: scf.yield
  tt.func public @loop_expand_axis1_spt4(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> {
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %zero = arith.constant dense<0.0> : tensor<512x16xf32, #blocked>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %colsf = arith.sitofp %cols : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols2 = tt.expand_dims %colsf {axis = 0 : i32} : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xf32, #blocked>
    %colsb = tt.broadcast %cols2 : tensor<1x16xf32, #blocked> -> tensor<512x16xf32, #blocked>
    %offs = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked1>
    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked1>
    %acc = scf.for %i = %c0_i32 to %n step %c512_i32 iter_args(%acc_i = %zero) -> (tensor<512x16xf32, #blocked>) : i32 {
      %is = tt.splat %i : i32 -> tensor<512xi32, #blocked1>
      %o = arith.addi %offs, %is : tensor<512xi32, #blocked1>
      %p = tt.addptr %base, %o : tensor<512x!tt.ptr<f32>, #blocked1>, tensor<512xi32, #blocked1>
      %x = tt.load %p : tensor<512x!tt.ptr<f32>, #blocked1>
      %xc = ttg.convert_layout %x : tensor<512xf32, #blocked1> -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %x2 = tt.expand_dims %xc {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<512x1xf32, #blocked>
      %xb = tt.broadcast %x2 : tensor<512x1xf32, #blocked> -> tensor<512x16xf32, #blocked>
      %prod = arith.mulf %xb, %colsb : tensor<512x16xf32, #blocked>
      %next = arith.addf %acc_i, %prod : tensor<512x16xf32, #blocked>
      scf.yield %next : tensor<512x16xf32, #blocked>
    }
    %sum = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<512x16xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return %sum : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
  }
}

// -----

// Negative 1: addmv anchored by a 2D LoadOp (models addmv_fp16_16x512_w4).
// The 2D load A anchors the loop accumulator in #blocked_a ([1, 8], wpc=[2, 2]).
// Moving acc onto the 1D load x's synthesized parent ([1, 4], wpc=[1, 4]) would
// force converting the 16x512 matrix A every iteration, so the pass keeps #blocked_a.
// The Windows RLC pipeline converts x while it is still f16.

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_x = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // RLC-LABEL: tt.func public @addmv_anchored_2d_load
  // RLC: scf.for {{.*}} -> (tensor<16x512xf32, #blocked>)
  // RLC: tt.load {{.*}} : tensor<16x512x!tt.ptr<f16>, #blocked>
  // RLC: ttg.convert_layout %{{.*}} : tensor<512xf16, #blocked1> -> tensor<512xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
  // RLC: scf.yield {{.*}} : tensor<16x512xf32, #blocked>

  // ALLOC-LABEL: tt.func public @addmv_anchored_2d_load
  // ALLOC: scf.for
  // ALLOC: ttg.convert_layout {{.*}} {allocation.offset = 0 : i32} : tensor<512xf16, #blocked1> -> tensor<512xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
  tt.func public @addmv_anchored_2d_load(%A_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                         %x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                         %K: i32) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked_a}>> {
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %zero = arith.constant dense<0.0> : tensor<16x512xf32, #blocked_a>
    %offs_m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked_a}>>
    %offs_m2 = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked_a}>> -> tensor<16x1xi32, #blocked_a>
    %stride_a = arith.constant dense<512> : tensor<16x1xi32, #blocked_a>
    %row_off = arith.muli %offs_m2, %stride_a : tensor<16x1xi32, #blocked_a>
    %a_base = tt.splat %A_ptr : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>, #blocked_a>
    %a_row = tt.addptr %a_base, %row_off : tensor<16x1x!tt.ptr<f16>, #blocked_a>, tensor<16x1xi32, #blocked_a>
    %a_row_b = tt.broadcast %a_row : tensor<16x1x!tt.ptr<f16>, #blocked_a> -> tensor<16x512x!tt.ptr<f16>, #blocked_a>
    %offs_ka = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %offs_ka2 = tt.expand_dims %offs_ka {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x512xi32, #blocked_a>
    %offs_kx = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked_x>
    %x_base = tt.splat %x_ptr : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked_x>
    %acc = scf.for %k = %c0_i32 to %K step %c512_i32 iter_args(%acc_i = %zero) -> (tensor<16x512xf32, #blocked_a>) : i32 {
      %ks_a = tt.splat %k : i32 -> tensor<1x512xi32, #blocked_a>
      %ak = arith.addi %ks_a, %offs_ka2 : tensor<1x512xi32, #blocked_a>
      %ak_b = tt.broadcast %ak : tensor<1x512xi32, #blocked_a> -> tensor<16x512xi32, #blocked_a>
      %ap = tt.addptr %a_row_b, %ak_b : tensor<16x512x!tt.ptr<f16>, #blocked_a>, tensor<16x512xi32, #blocked_a>
      %a16 = tt.load %ap : tensor<16x512x!tt.ptr<f16>, #blocked_a>
      %a32 = arith.extf %a16 : tensor<16x512xf16, #blocked_a> to tensor<16x512xf32, #blocked_a>
      %ks_x = tt.splat %k : i32 -> tensor<512xi32, #blocked_x>
      %xk = arith.addi %ks_x, %offs_kx : tensor<512xi32, #blocked_x>
      %xp = tt.addptr %x_base, %xk : tensor<512x!tt.ptr<f16>, #blocked_x>, tensor<512xi32, #blocked_x>
      %x16 = tt.load %xp : tensor<512x!tt.ptr<f16>, #blocked_x>
      %x32 = arith.extf %x16 : tensor<512xf16, #blocked_x> to tensor<512xf32, #blocked_x>
      %xc = ttg.convert_layout %x32 : tensor<512xf32, #blocked_x> -> tensor<512xf32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
      %x2 = tt.expand_dims %xc {axis = 0 : i32} : tensor<512xf32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x512xf32, #blocked_a>
      %xb = tt.broadcast %x2 : tensor<1x512xf32, #blocked_a> -> tensor<16x512xf32, #blocked_a>
      %prod = arith.mulf %a32, %xb : tensor<16x512xf32, #blocked_a>
      %next = arith.addf %acc_i, %prod : tensor<16x512xf32, #blocked_a>
      scf.yield %next : tensor<16x512xf32, #blocked_a>
    }
    %y = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x512xf32, #blocked_a>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked_a}>>
    tt.return %y : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked_a}>>
  }
}

// -----

// Negative 2: External user requires the full 2D accumulator in its original
// layout (no post-loop reduce), and converting the 16x1024 result outside the
// loop costs more than the 1024-element in-loop convert (1 * 65536 > 8 * 2048).
// The pass rejects the synthesized parent because total cost would increase.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // RLC-LABEL: tt.func public @loop_external_user_keeps_original
  // RLC: scf.for {{.*}} -> (tensor<16x1024xf32, #blocked>)
  // RLC: ttg.convert_layout %{{.*}} : tensor<1024xf16, #blocked1> -> tensor<1024xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
  // RLC: scf.yield {{.*}} : tensor<16x1024xf32, #blocked>
  // RLC: tt.return %{{.*}} : tensor<16x1024xf32, #blocked>

  // ALLOC-LABEL: tt.func public @loop_external_user_keeps_original
  // ALLOC: scf.for
  // ALLOC: ttg.convert_layout {{.*}} {allocation.offset = 0 : i32} : tensor<1024xf16, #blocked1> -> tensor<1024xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
  tt.func public @loop_external_user_keeps_original(%ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %n: i32) -> tensor<16x1024xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %zero = arith.constant dense<0.0> : tensor<16x1024xf32, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %base = tt.splat %ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1>
    %acc = scf.for %i = %c0_i32 to %n step %c1024_i32 iter_args(%acc_i = %zero) -> (tensor<16x1024xf32, #blocked>) : i32 {
      %is = tt.splat %i : i32 -> tensor<1024xi32, #blocked1>
      %o = arith.addi %offs, %is : tensor<1024xi32, #blocked1>
      %p = tt.addptr %base, %o : tensor<1024x!tt.ptr<f16>, #blocked1>, tensor<1024xi32, #blocked1>
      %x = tt.load %p : tensor<1024x!tt.ptr<f16>, #blocked1>
      %xc = ttg.convert_layout %x : tensor<1024xf16, #blocked1> -> tensor<1024xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
      %xf = arith.extf %xc : tensor<1024xf16, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<1024xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %x2 = tt.expand_dims %xf {axis = 0 : i32} : tensor<1024xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xf32, #blocked>
      %xb = tt.broadcast %x2 : tensor<1x1024xf32, #blocked> -> tensor<16x1024xf32, #blocked>
      %next = arith.addf %acc_i, %xb : tensor<16x1024xf32, #blocked>
      scf.yield %next : tensor<16x1024xf32, #blocked>
    }
    tt.return %acc : tensor<16x1024xf32, #blocked>
  }
}

// -----

// Boundary test: scf.for carrying two tensors where only the first moves onto
// the synthesized parent layout and the second stays in #blocked, checking
// consistent init/iter_arg/yield/result types across the loop boundary.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // RLC-DAG: #[[$P2:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
  // RLC-LABEL: tt.func public @loop_multi_iter_arg_boundary
  // RLC: scf.for {{.*}} -> (tensor<16x1024xf32, #[[$P2]]>, tensor<16x1024xf32, #blocked>)
  // RLC: scf.yield {{.*}} : tensor<16x1024xf32, #[[$P2]]>, tensor<16x1024xf32, #blocked>
  // RLC: tt.return %{{.*}}, %{{.*}} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16x1024xf32, #blocked>

  // ALLOC-LABEL: tt.func public @loop_multi_iter_arg_boundary
  // ALLOC: scf.for
  // ALLOC-NOT: allocation.offset
  // ALLOC: scf.yield
  tt.func public @loop_multi_iter_arg_boundary(%ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                               %keep_init: tensor<16x1024xf32, #blocked>,
                                               %n: i32) -> (tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16x1024xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %zero = arith.constant dense<0.0> : tensor<16x1024xf32, #blocked>
    %one = arith.constant dense<1.0> : tensor<16x1024xf32, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %base = tt.splat %ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1>
    %res:2 = scf.for %i = %c0_i32 to %n step %c1024_i32 iter_args(%acc_i = %zero, %keep_i = %keep_init) -> (tensor<16x1024xf32, #blocked>, tensor<16x1024xf32, #blocked>) : i32 {
      %is = tt.splat %i : i32 -> tensor<1024xi32, #blocked1>
      %o = arith.addi %offs, %is : tensor<1024xi32, #blocked1>
      %p = tt.addptr %base, %o : tensor<1024x!tt.ptr<f16>, #blocked1>, tensor<1024xi32, #blocked1>
      %x = tt.load %p : tensor<1024x!tt.ptr<f16>, #blocked1>
      %xc = ttg.convert_layout %x : tensor<1024xf16, #blocked1> -> tensor<1024xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
      %xf = arith.extf %xc : tensor<1024xf16, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<1024xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %x2 = tt.expand_dims %xf {axis = 0 : i32} : tensor<1024xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xf32, #blocked>
      %xb = tt.broadcast %x2 : tensor<1x1024xf32, #blocked> -> tensor<16x1024xf32, #blocked>
      %next = arith.addf %acc_i, %xb : tensor<16x1024xf32, #blocked>
      %keep_next = arith.addf %keep_i, %one : tensor<16x1024xf32, #blocked>
      scf.yield %next, %keep_next : tensor<16x1024xf32, #blocked>, tensor<16x1024xf32, #blocked>
    }
    %sum = "tt.reduce"(%res#0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x1024xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %sum, %res#1 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16x1024xf32, #blocked>
  }
}

// -----

// Target gate regression: SM80 (cuda:80) must NOT activate the SM75-only
// expand_dims parent synthesis or loop-component resolution.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // RLC-LABEL: tt.func public @loop_histogram_sm80_unchanged
  // RLC: scf.for {{.*}} -> (tensor<16x1024xi64, #blocked>)
  // RLC: ttg.convert_layout %{{.*}} : tensor<1024xi16, #blocked1> -> tensor<1024xi16, #ttg.slice<{dim = 0, parent = #blocked}>>
  // RLC: scf.yield {{.*}} : tensor<16x1024xi64, #blocked>

  // ALLOC: module attributes {{{.*}}ttg.shared = 2048 : i32
  // ALLOC-LABEL: tt.func public @loop_histogram_sm80_unchanged
  // ALLOC: scf.for
  // ALLOC: ttg.convert_layout {{.*}} {allocation.offset = 0 : i32} : tensor<1024xi16, #blocked1> -> tensor<1024xi16, #ttg.slice<{dim = 0, parent = #blocked}>>
  tt.func public @loop_histogram_sm80_unchanged(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %n: i32) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %zero = arith.constant dense<0> : tensor<16x1024xi64, #blocked>
    %fifteen = arith.constant dense<15> : tensor<1024xi16, #blocked1>
    %bins = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bins2 = tt.expand_dims %bins {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %binsb = tt.broadcast %bins2 : tensor<16x1xi32, #blocked> -> tensor<16x1024xi32, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %base = tt.splat %ptr : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked1>
    %acc = scf.for %i = %c0_i32 to %n step %c1024_i32 iter_args(%acc_i = %zero) -> (tensor<16x1024xi64, #blocked>) : i32 {
      %is = tt.splat %i : i32 -> tensor<1024xi32, #blocked1>
      %o = arith.addi %offs, %is : tensor<1024xi32, #blocked1>
      %p = tt.addptr %base, %o : tensor<1024x!tt.ptr<i16>, #blocked1>, tensor<1024xi32, #blocked1>
      %x = tt.load %p : tensor<1024x!tt.ptr<i16>, #blocked1>
      %key = arith.andi %x, %fifteen : tensor<1024xi16, #blocked1>
      %keyc = ttg.convert_layout %key : tensor<1024xi16, #blocked1> -> tensor<1024xi16, #ttg.slice<{dim = 0, parent = #blocked}>>
      %keyi = arith.extui %keyc : tensor<1024xi16, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %key2 = tt.expand_dims %keyi {axis = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xi32, #blocked>
      %keyb = tt.broadcast %key2 : tensor<1x1024xi32, #blocked> -> tensor<16x1024xi32, #blocked>
      %hit = arith.cmpi eq, %binsb, %keyb : tensor<16x1024xi32, #blocked>
      %hit64 = arith.extui %hit : tensor<16x1024xi1, #blocked> to tensor<16x1024xi64, #blocked>
      %next = arith.addi %acc_i, %hit64 : tensor<16x1024xi64, #blocked>
      scf.yield %next : tensor<16x1024xi64, #blocked>
    }
    %count = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
    ^bb0(%a: i64, %b: i64):
      %s = arith.addi %a, %b : i64
      tt.reduce.return %s : i64
    }) : (tensor<16x1024xi64, #blocked>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %count : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Control-flow regression: nested scf.while inside scf.for where scf.yield
// carries more values (beforeArguments: i32, tensor) than scf.while results
// (only i32 via scf.condition). Previously triggered getResult out-of-bounds.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // RLC-LABEL: tt.func public @nested_while_fewer_results
  // RLC: scf.for
  // RLC: scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, tensor<16x1024xf16, #{{.*}}>) -> i32
  // RLC: scf.condition(%{{.*}}) %{{.*}} : i32
  // RLC: scf.yield %{{.*}}, %{{.*}} : i32, tensor<16x1024xf16, #{{.*}}>

  // ALLOC-LABEL: tt.func public @nested_while_fewer_results
  // ALLOC: scf.while
  tt.func public @nested_while_fewer_results(%ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %n: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst0 = arith.constant 0.000000e+00 : f32
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %base = tt.splat %ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1>
    %total = scf.for %i = %c0_i32 to %n step %c1024_i32 iter_args(%tot_i = %c0_i32) -> (i32) : i32 {
      %is = tt.splat %i : i32 -> tensor<1024xi32, #blocked1>
      %o = arith.addi %offs, %is : tensor<1024xi32, #blocked1>
      %p = tt.addptr %base, %o : tensor<1024x!tt.ptr<f16>, #blocked1>, tensor<1024xi32, #blocked1>
      %x = tt.load %p : tensor<1024x!tt.ptr<f16>, #blocked1>
      %xc = ttg.convert_layout %x : tensor<1024xf16, #blocked1> -> tensor<1024xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
      %x2 = tt.expand_dims %xc {axis = 0 : i32} : tensor<1024xf16, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xf16, #blocked>
      %xb = tt.broadcast %x2 : tensor<1x1024xf16, #blocked> -> tensor<16x1024xf16, #blocked>
      %cnt_out = scf.while (%cnt = %c0_i32, %acc = %xb) : (i32, tensor<16x1024xf16, #blocked>) -> i32 {
        %lt = arith.cmpi slt, %cnt, %c4_i32 : i32
        %flat = tt.reshape %acc allow_reorder : tensor<16x1024xf16, #blocked> -> tensor<16384xf16, #blocked1>
        %red = "tt.reduce"(%flat) <{axis = 0 : i32}> ({
        ^bb0(%a: f16, %b: f16):
          %s = arith.addf %a, %b : f16
          tt.reduce.return %s : f16
        }) : (tensor<16384xf16, #blocked1>) -> f16
        %red32 = arith.extf %red : f16 to f32
        %pos = arith.cmpf ogt, %red32, %cst0 : f32
        %cond = arith.andi %lt, %pos : i1
        scf.condition(%cond) %cnt : i32
      } do {
      ^bb0(%cnt_d: i32):
        %cnt_n = arith.addi %cnt_d, %c1_i32 : i32
        %cnt_f = arith.sitofp %cnt_n : i32 to f16
        %cnt_s = tt.splat %cnt_f : f16 -> tensor<1x1024xf16, #blocked>
        %sub = arith.subf %x2, %cnt_s : tensor<1x1024xf16, #blocked>
        %acc_n = tt.broadcast %sub : tensor<1x1024xf16, #blocked> -> tensor<16x1024xf16, #blocked>
        scf.yield %cnt_n, %acc_n : i32, tensor<16x1024xf16, #blocked>
      }
      %tot_n = arith.addi %tot_i, %cnt_out : i32
      scf.yield %tot_n : i32
    }
    tt.return %total : i32
  }
}
