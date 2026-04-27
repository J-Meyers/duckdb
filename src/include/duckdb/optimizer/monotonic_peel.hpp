//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/monotonic_peel.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb/common/common.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

struct MonotonicPeelStep {
	idx_t col_arg;
	bool inverts;
};

//! Inspect ONE wrapper level of `expr`. On success, `step.col_arg` is the index of the
//! column-bearing child to walk into next, and `step.inverts` is true iff this hop
//! reverses ordering. Handles invertible casts (single child, never inverts), commutative
//! `+` (column on either side), and any function declaring FunctionMonotonicity with all
//! other args foldable.
//!
//! allow_finite_only=false (Tier-1): refuses functions whose annotation has
//!   requires_finite_input=true. Use this for the structural plan-time peel — safe only
//!   when the function is total over its entire input domain.
//! allow_finite_only=true (Tier-2): accepts requires_finite_input functions. The caller
//!   is responsible for a runtime safety gate (e.g. checking the folded value is non-NULL
//!   before committing the result).
bool TryPeelMonotonicLevel(const Expression &expr, MonotonicPeelStep &step, bool allow_finite_only = false);

} // namespace duckdb
