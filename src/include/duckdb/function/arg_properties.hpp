//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/arg_properties.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

class BoundFunctionExpression;

//! Per-argument monotonicity (other args held constant).
enum class Monotonicity : uint8_t {
	UNKNOWN = 0,
	CONSTANT,
	NON_DECREASING,
	STRICTLY_INCREASING,
	NON_INCREASING,
	STRICTLY_DECREASING,
};

//! Half-open interval [lo, hi).
struct HalfOpenInterval {
	Value lo;
	Value hi;
};

//! Declarative metadata for one argument of a scalar function.
struct ArgProperties {
	Monotonicity monotonicity = Monotonicity::UNKNOWN;
	//! True if the function may return NULL on +/-inf or NaN inputs.
	bool requires_finite_input = false;
	//! Distinct inputs map to distinct outputs.
	bool injective = false;

	//! Refines monotonicity given the actual value range of this arg.
	using MonotonicityRefiner = Monotonicity (*)(const BoundFunctionExpression &expr, idx_t arg_idx, const Value &lo,
	                                             const Value &hi);
	MonotonicityRefiner refine_monotonicity = nullptr;

	//! Input intervals corresponding to a single output point; empty = unsatisfiable.
	using PreimageFn = vector<HalfOpenInterval> (*)(const BoundFunctionExpression &expr, idx_t arg_idx,
	                                                const Value &output_point);
	PreimageFn preimage = nullptr;

	//! Strict monotonicity in this arg implies injectivity in this arg (held-other-args fixed):
	//! a < b => f(a) < f(b) means f never collapses distinct sources, so it's order-preserving and
	//! one-to-one. Set both fields when strict so consumers like distinct-count propagation see
	//! the truth without each call site having to remember to chain `.Injective()`.
	ArgProperties &Increasing(bool strict = true) {
		monotonicity = strict ? Monotonicity::STRICTLY_INCREASING : Monotonicity::NON_DECREASING;
		if (strict) {
			injective = true;
		}
		return *this;
	}
	ArgProperties &Decreasing(bool strict = true) {
		monotonicity = strict ? Monotonicity::STRICTLY_DECREASING : Monotonicity::NON_INCREASING;
		if (strict) {
			injective = true;
		}
		return *this;
	}
	ArgProperties &Constant() {
		monotonicity = Monotonicity::CONSTANT;
		return *this;
	}
	ArgProperties &RequiresFinite() {
		requires_finite_input = true;
		return *this;
	}
	ArgProperties &Injective() {
		injective = true;
		return *this;
	}
	ArgProperties &WithMonotonicityRefiner(MonotonicityRefiner r) {
		refine_monotonicity = r;
		return *this;
	}
	ArgProperties &WithPreimage(PreimageFn p) {
		preimage = p;
		return *this;
	}
};

constexpr bool IsMonotonicIncreasing(Monotonicity m) {
	return m == Monotonicity::NON_DECREASING || m == Monotonicity::STRICTLY_INCREASING;
}
constexpr bool IsMonotonicDecreasing(Monotonicity m) {
	return m == Monotonicity::NON_INCREASING || m == Monotonicity::STRICTLY_DECREASING;
}
constexpr bool IsKnownMonotonic(Monotonicity m) {
	return m != Monotonicity::UNKNOWN;
}
constexpr bool IsStrict(Monotonicity m) {
	return m == Monotonicity::STRICTLY_INCREASING || m == Monotonicity::STRICTLY_DECREASING;
}

} // namespace duckdb
