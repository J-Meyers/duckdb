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

//! Monotonicity of a function in one argument (other args held constant).
enum class Monotonicity : uint8_t {
	UNKNOWN = 0,
	CONSTANT,
	NON_DECREASING,
	STRICTLY_INCREASING,
	NON_INCREASING,
	STRICTLY_DECREASING,
};

//! Interval [lo, hi) — lo inclusive, hi exclusive.
struct HalfOpenInterval {
	Value lo;
	Value hi;
};

struct ArgProperties {
	Monotonicity monotonicity = Monotonicity::UNKNOWN;
	//! May return NULL on +/-inf or NaN inputs.
	bool requires_finite_input = false;
	bool injective = false;

	//! Inputs mapping to `output_point`. Empty = nothing maps to it.
	using PreimageFn = vector<HalfOpenInterval> (*)(const BoundFunctionExpression &expr, idx_t arg_idx,
	                                                const Value &output_point);
	PreimageFn preimage = nullptr;

	ArgProperties &StrictlyIncreasing() {
		monotonicity = Monotonicity::STRICTLY_INCREASING;
		return *this;
	}
	ArgProperties &NonDecreasing() {
		monotonicity = Monotonicity::NON_DECREASING;
		return *this;
	}
	ArgProperties &StrictlyDecreasing() {
		monotonicity = Monotonicity::STRICTLY_DECREASING;
		return *this;
	}
	ArgProperties &NonIncreasing() {
		monotonicity = Monotonicity::NON_INCREASING;
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
