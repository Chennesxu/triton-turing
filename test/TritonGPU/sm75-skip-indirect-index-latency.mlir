// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies=num-stages=3 | FileCheck %s --check-prefix=LAT
// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies=num-stages=3 -tritongpu-schedule-loops -tritongpu-pipeline=num-stages=3 | FileCheck %s --check-prefix=PIPE

// Positive: SM75, num_stages=3, clamped to 1 slot, dist==1 integer index load
// feeding only downstream dist==0 load address calculation. The index load gets
// no tt.latency and no shared buffer, while the main data loads stay at 1 slot.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // LAT: module attributes {{{.*}}ttg.sm75_pipeline_slots = 1 : i32, ttg.sm75_pipeline_slots_requested = 1 : i32
  // LAT-LABEL: tt.func public @sm75_s3_skip_indirect_index
  // LAT: scf.for
  // LAT: tt.load %{{.*}} : tensor<32x!tt.ptr<i32>
  // LAT-NOT: tt.latency
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<64x32x!tt.ptr<f16>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<32x64x!tt.ptr<f16>

  // PIPE-LABEL: tt.func public @sm75_s3_skip_indirect_index
  // PIPE-NOT: memdesc<1x32xi32
  // PIPE: ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16
  // PIPE: ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16
  // PIPE-NOT: memdesc<1x32xi32
  tt.func public @sm75_s3_skip_indirect_index(
      %a_base: tensor<64x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>},
      %b_base: tensor<32x1x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 1]> : tensor<2xi32>},
      %idx_base: tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<32> : tensor<1xi32>},
      %stride_b: tensor<32x1xi64, #blocked1>,
      %cols_b: tensor<32x64xi32, #blocked1> {tt.divisibility = dense<[1, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>},
      %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<64x64xf32, #mma>
    %a_step = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %idx_step = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %res:3 = scf.for %iv = %c0 to %n step %c32 iter_args(%acc = %cst, %ap = %a_base, %ip = %idx_base)
        -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>) : i32 {
      %idx = tt.load %ip : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx64 = arith.extsi %idx : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx2 = tt.expand_dims %idx64 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi64, #blocked1>
      %row_off = arith.muli %idx2, %stride_b : tensor<32x1xi64, #blocked1>
      %b_row = tt.addptr %b_base, %row_off : tensor<32x1x!tt.ptr<f16>, #blocked1>, tensor<32x1xi64, #blocked1>
      %b_bcast = tt.broadcast %b_row : tensor<32x1x!tt.ptr<f16>, #blocked1> -> tensor<32x64x!tt.ptr<f16>, #blocked1>
      %bp = tt.addptr %b_bcast, %cols_b : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
      %a = tt.load %ap : tensor<64x32x!tt.ptr<f16>, #blocked>
      %b = tt.load %bp : tensor<32x64x!tt.ptr<f16>, #blocked1>
      %ad = ttg.convert_layout %a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %bd = ttg.convert_layout %b : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %next_acc = tt.dot %ad, %bd, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      %next_ap = tt.addptr %ap, %a_step : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %next_ip = tt.addptr %ip, %idx_step : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      scf.yield %next_acc, %next_ap, %next_ip : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    }
    tt.return %res#0 : tensor<64x64xf32, #mma>
  }
}

// -----

// Negative 1: dist==1 FP16 load used as `other` padding value of a dist==0 load
// (matches other_operand.py). Must keep tt.latency = 1.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // LAT-LABEL: tt.func public @sm75_s3_keep_other_operand
  // LAT: scf.for
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<64x!tt.ptr<f16>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<32x32x!tt.ptr<f16>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<32x64x!tt.ptr<f16>

  // PIPE-LABEL: tt.func public @sm75_s3_keep_other_operand
  // PIPE: ttg.local_alloc : () -> !ttg.memdesc<1x64xf16
  tt.func public @sm75_s3_keep_other_operand(
      %a_base: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>},
      %b_base: tensor<32x64x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>},
      %pad_base: tensor<64x!tt.ptr<f16>, #ttg.slice<{dim = 0, parent = #blocked1}>> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<64> : tensor<1xi32>},
      %b_mask: tensor<32x64xi1, #blocked1>,
      %n: i32) -> tensor<32x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<32x64xf32, #mma>
    %a_step = arith.constant dense<32> : tensor<32x32xi32, #blocked>
    %b_step = arith.constant dense<2048> : tensor<32x64xi32, #blocked1>
    %pad_step = arith.constant dense<64> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %res:4 = scf.for %iv = %c0 to %n step %c32 iter_args(%acc = %cst, %ap = %a_base, %bp = %b_base, %pp = %pad_base)
        -> (tensor<32x64xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<64x!tt.ptr<f16>, #ttg.slice<{dim = 0, parent = #blocked1}>>) : i32 {
      %filler = tt.load %pp : tensor<64x!tt.ptr<f16>, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %filler2 = tt.expand_dims %filler {axis = 0 : i32} : tensor<64xf16, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xf16, #blocked1>
      %other = tt.broadcast %filler2 : tensor<1x64xf16, #blocked1> -> tensor<32x64xf16, #blocked1>
      %a = tt.load %ap : tensor<32x32x!tt.ptr<f16>, #blocked>
      %b = tt.load %bp, %b_mask, %other : tensor<32x64x!tt.ptr<f16>, #blocked1>
      %ad = ttg.convert_layout %a : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %bd = ttg.convert_layout %b : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %next_acc = tt.dot %ad, %bd, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x64xf32, #mma>
      %next_ap = tt.addptr %ap, %a_step : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %next_bp = tt.addptr %bp, %b_step : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
      %next_pp = tt.addptr %pp, %pad_step : tensor<64x!tt.ptr<f16>, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      scf.yield %next_acc, %next_ap, %next_bp, %next_pp : tensor<32x64xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<64x!tt.ptr<f16>, #ttg.slice<{dim = 0, parent = #blocked1}>>
    }
    tt.return %res#0 : tensor<32x64xf32, #mma>
  }
}

// -----

// Negative 2: dist==1 integer load feeds address calculation AND ALSO feeds a
// mask comparison (extra non-address consumer). Must keep tt.latency = 1.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // LAT-LABEL: tt.func public @sm75_s3_keep_index_with_mask_use
  // LAT: scf.for
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<32x!tt.ptr<i32>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<64x32x!tt.ptr<f16>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<32x64x!tt.ptr<f16>

  // PIPE-LABEL: tt.func public @sm75_s3_keep_index_with_mask_use
  // PIPE: ttg.local_alloc : () -> !ttg.memdesc<1x32xi32
  tt.func public @sm75_s3_keep_index_with_mask_use(
      %a_base: tensor<64x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>},
      %b_base: tensor<32x1x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 1]> : tensor<2xi32>},
      %idx_base: tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<32> : tensor<1xi32>},
      %stride_b: tensor<32x1xi64, #blocked1>,
      %cols_b: tensor<32x64xi32, #blocked1> {tt.divisibility = dense<[1, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>},
      %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<64x64xf32, #mma>
    %zero_i32 = arith.constant dense<0> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %zero_f16 = arith.constant dense<0.0> : tensor<32x64xf16, #blocked1>
    %a_step = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %idx_step = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %res:3 = scf.for %iv = %c0 to %n step %c32 iter_args(%acc = %cst, %ap = %a_base, %ip = %idx_base)
        -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>) : i32 {
      %idx = tt.load %ip : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %valid = arith.cmpi sge, %idx, %zero_i32 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %valid2 = tt.expand_dims %valid {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi1, #blocked1>
      %b_mask = tt.broadcast %valid2 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
      %idx64 = arith.extsi %idx : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx2 = tt.expand_dims %idx64 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi64, #blocked1>
      %row_off = arith.muli %idx2, %stride_b : tensor<32x1xi64, #blocked1>
      %b_row = tt.addptr %b_base, %row_off : tensor<32x1x!tt.ptr<f16>, #blocked1>, tensor<32x1xi64, #blocked1>
      %b_bcast = tt.broadcast %b_row : tensor<32x1x!tt.ptr<f16>, #blocked1> -> tensor<32x64x!tt.ptr<f16>, #blocked1>
      %bp = tt.addptr %b_bcast, %cols_b : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
      %a = tt.load %ap : tensor<64x32x!tt.ptr<f16>, #blocked>
      %b = tt.load %bp, %b_mask, %zero_f16 : tensor<32x64x!tt.ptr<f16>, #blocked1>
      %ad = ttg.convert_layout %a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %bd = ttg.convert_layout %b : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %next_acc = tt.dot %ad, %bd, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      %next_ap = tt.addptr %ap, %a_step : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %next_ip = tt.addptr %ip, %idx_step : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      scf.yield %next_acc, %next_ap, %next_ip : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    }
    tt.return %res#0 : tensor<64x64xf32, #mma>
  }
}

// -----

// Negative 3: SM80 (cuda:80) must NOT activate the SM75-only rule.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // LAT-LABEL: tt.func public @sm80_s3_keep_indirect_index
  // LAT: scf.for
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<32x!tt.ptr<i32>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<64x32x!tt.ptr<f16>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<32x64x!tt.ptr<f16>

  // PIPE-LABEL: tt.func public @sm80_s3_keep_indirect_index
  // PIPE: ttg.local_alloc : () -> !ttg.memdesc<1x32xi32
  tt.func public @sm80_s3_keep_indirect_index(
      %a_base: tensor<64x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>},
      %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %idx_base: tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<32> : tensor<1xi32>},
      %stride_b_i32: i32 {tt.divisibility = 16 : i32},
      %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<64x64xf32, #mma>
    %a_step = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %idx_step = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %stride_b64 = arith.extsi %stride_b_i32 : i32 to i64
    %stride_b = tt.splat %stride_b64 : i64 -> tensor<32x1xi64, #blocked1>
    %b_base = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked1>
    %cols1d = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cols2d = tt.expand_dims %cols1d {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %cols_b = tt.broadcast %cols2d : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %res:3 = scf.for %iv = %c0 to %n step %c32 iter_args(%acc = %cst, %ap = %a_base, %ip = %idx_base)
        -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>) : i32 {
      %idx = tt.load %ip : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx64 = arith.extsi %idx : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx2 = tt.expand_dims %idx64 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi64, #blocked1>
      %row_off = arith.muli %idx2, %stride_b : tensor<32x1xi64, #blocked1>
      %b_row = tt.addptr %b_base, %row_off : tensor<32x1x!tt.ptr<f16>, #blocked1>, tensor<32x1xi64, #blocked1>
      %b_bcast = tt.broadcast %b_row : tensor<32x1x!tt.ptr<f16>, #blocked1> -> tensor<32x64x!tt.ptr<f16>, #blocked1>
      %bp = tt.addptr %b_bcast, %cols_b : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
      %a = tt.load %ap : tensor<64x32x!tt.ptr<f16>, #blocked>
      %b = tt.load %bp : tensor<32x64x!tt.ptr<f16>, #blocked1>
      %ad = ttg.convert_layout %a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %bd = ttg.convert_layout %b : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %next_acc = tt.dot %ad, %bd, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      %next_ap = tt.addptr %ap, %a_step : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %next_ip = tt.addptr %ip, %idx_step : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      scf.yield %next_acc, %next_ap, %next_ip : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    }
    tt.return %res#0 : tensor<64x64xf32, #mma>
  }
}

// -----

// Negative 4: User-provided tt.latency must be preserved verbatim on SM75.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // LAT-LABEL: tt.func public @sm75_s3_preserve_user_latency
  // LAT: scf.for
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<32x!tt.ptr<i32>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<64x32x!tt.ptr<f16>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<32x64x!tt.ptr<f16>

  // PIPE-LABEL: tt.func public @sm75_s3_preserve_user_latency
  // PIPE: ttg.local_alloc : () -> !ttg.memdesc<1x32xi32
  tt.func public @sm75_s3_preserve_user_latency(
      %a_base: tensor<64x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>},
      %b_base: tensor<32x1x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 1]> : tensor<2xi32>},
      %idx_base: tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<32> : tensor<1xi32>},
      %stride_b: tensor<32x1xi64, #blocked1>,
      %cols_b: tensor<32x64xi32, #blocked1> {tt.divisibility = dense<[1, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>},
      %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<64x64xf32, #mma>
    %a_step = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %idx_step = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %res:3 = scf.for %iv = %c0 to %n step %c32 iter_args(%acc = %cst, %ap = %a_base, %ip = %idx_base)
        -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>) : i32 {
      %idx = tt.load %ip {tt.latency = 1 : i32} : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx64 = arith.extsi %idx : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx2 = tt.expand_dims %idx64 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi64, #blocked1>
      %row_off = arith.muli %idx2, %stride_b : tensor<32x1xi64, #blocked1>
      %b_row = tt.addptr %b_base, %row_off : tensor<32x1x!tt.ptr<f16>, #blocked1>, tensor<32x1xi64, #blocked1>
      %b_bcast = tt.broadcast %b_row : tensor<32x1x!tt.ptr<f16>, #blocked1> -> tensor<32x64x!tt.ptr<f16>, #blocked1>
      %bp = tt.addptr %b_bcast, %cols_b : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
      %a = tt.load %ap {tt.latency = 1 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked>
      %b = tt.load %bp {tt.latency = 1 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked1>
      %ad = ttg.convert_layout %a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %bd = ttg.convert_layout %b : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %next_acc = tt.dot %ad, %bd, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      %next_ap = tt.addptr %ap, %a_step : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %next_ip = tt.addptr %ip, %idx_step : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      scf.yield %next_acc, %next_ap, %next_ip : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    }
    tt.return %res#0 : tensor<64x64xf32, #mma>
  }
}

// -----

// Negative 5: SM75 s3 boundary where full-set clamp produces 0 slots
// (128x128 f16 + 128x128 f16 + 128 i32 = 66048 B > 65536 B). Must stay at
// 0 slots and NOT promote 0 -> 1 by subtracting index bytes before clamping.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [4, 2], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // LAT: module attributes {{{.*}}ttg.sm75_pipeline_slots = 0 : i32, ttg.sm75_pipeline_slots_requested = 1 : i32
  // LAT-LABEL: tt.func public @sm75_s3_clamp_zero_boundary
  // LAT: scf.for
  // LAT-NOT: tt.latency
  // LAT: scf.yield

  // PIPE-LABEL: tt.func public @sm75_s3_clamp_zero_boundary
  // PIPE-NOT: ttg.local_alloc
  tt.func public @sm75_s3_clamp_zero_boundary(
      %a_base: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 128]> : tensor<2xi32>},
      %b_base: tensor<128x1x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 1]> : tensor<2xi32>},
      %idx_base: tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<128> : tensor<1xi32>},
      %stride_b: tensor<128x1xi64, #blocked1>,
      %cols_b: tensor<128x128xi32, #blocked1> {tt.divisibility = dense<[1, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 128]> : tensor<2xi32>},
      %n: i32) -> tensor<128x128xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %cst = arith.constant dense<0.0> : tensor<128x128xf32, #mma>
    %a_step = arith.constant dense<128> : tensor<128x128xi32, #blocked>
    %idx_step = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %res:3 = scf.for %iv = %c0 to %n step %c128 iter_args(%acc = %cst, %ap = %a_base, %ip = %idx_base)
        -> (tensor<128x128xf32, #mma>, tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>) : i32 {
      %idx = tt.load %ip : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx64 = arith.extsi %idx : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx2 = tt.expand_dims %idx64 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi64, #blocked1>
      %row_off = arith.muli %idx2, %stride_b : tensor<128x1xi64, #blocked1>
      %b_row = tt.addptr %b_base, %row_off : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi64, #blocked1>
      %b_bcast = tt.broadcast %b_row : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
      %bp = tt.addptr %b_bcast, %cols_b : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
      %a = tt.load %ap : tensor<128x128x!tt.ptr<f16>, #blocked>
      %b = tt.load %bp : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %ad = ttg.convert_layout %a : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %bd = ttg.convert_layout %b : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %next_acc = tt.dot %ad, %bd, %acc : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %next_ap = tt.addptr %ap, %a_step : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
      %next_ip = tt.addptr %ip, %idx_step : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      scf.yield %next_acc, %next_ap, %next_ip : tensor<128x128xf32, #mma>, tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    }
    tt.return %res#0 : tensor<128x128xf32, #mma>
  }
}

// -----

// Negative 6: SM75 loop explicitly requesting tt.num_stages = 5 (clamped to
// 1 slot: 128x64 f16 + 64x128 f16 + 64 i32 = 33024 B/slot). Must NOT skip
// index load or bump slots 1 -> 2 because loop num_stages != 3.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [4, 2], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // LAT: module attributes {{{.*}}ttg.sm75_pipeline_slots = 1 : i32, ttg.sm75_pipeline_slots_requested = 2 : i32
  // LAT-LABEL: tt.func public @sm75_s5_keep_indirect_index
  // LAT: scf.for
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<64x!tt.ptr<i32>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<128x64x!tt.ptr<f16>
  // LAT: tt.load %{{.*}} {tt.latency = 1 : i32} : tensor<64x128x!tt.ptr<f16>

  // PIPE-LABEL: tt.func public @sm75_s5_keep_indirect_index
  // PIPE: ttg.local_alloc : () -> !ttg.memdesc<1x64xi32
  tt.func public @sm75_s5_keep_indirect_index(
      %a_base: tensor<128x64x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>},
      %b_base: tensor<64x1x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 1]> : tensor<2xi32>},
      %idx_base: tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<64> : tensor<1xi32>},
      %stride_b: tensor<64x1xi64, #blocked1>,
      %cols_b: tensor<64x128xi32, #blocked1> {tt.divisibility = dense<[1, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 128]> : tensor<2xi32>},
      %n: i32) -> tensor<128x128xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.0> : tensor<128x128xf32, #mma>
    %a_step = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %idx_step = arith.constant dense<64> : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %res:3 = scf.for %iv = %c0 to %n step %c64 iter_args(%acc = %cst, %ap = %a_base, %ip = %idx_base)
        -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>) : i32 {
      %idx = tt.load %ip : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx64 = arith.extsi %idx : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %idx2 = tt.expand_dims %idx64 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi64, #blocked1>
      %row_off = arith.muli %idx2, %stride_b : tensor<64x1xi64, #blocked1>
      %b_row = tt.addptr %b_base, %row_off : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi64, #blocked1>
      %b_bcast = tt.broadcast %b_row : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
      %bp = tt.addptr %b_bcast, %cols_b : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
      %a = tt.load %ap : tensor<128x64x!tt.ptr<f16>, #blocked>
      %b = tt.load %bp : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %ad = ttg.convert_layout %a : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %bd = ttg.convert_layout %b : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %next_acc = tt.dot %ad, %bd, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %next_ap = tt.addptr %ap, %a_step : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
      %next_ip = tt.addptr %ip, %idx_step : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      scf.yield %next_acc, %next_ap, %next_ip : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    } {tt.num_stages = 5 : i32}
    tt.return %res#0 : tensor<128x128xf32, #mma>
  }
}
