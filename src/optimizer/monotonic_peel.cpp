#include "duckdb/optimizer/monotonic_peel.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

bool TryPeelMonotonicLevel(const Expression &expr, MonotonicPeelStep &step, bool allow_finite_only) {
	step.inverts = false;
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_CAST) {
		auto &cast = expr.Cast<BoundCastExpression>();
		if (!BoundCastExpression::CastIsInvertible(cast.child->return_type, cast.return_type)) {
			return false;
		}
		step.col_arg = 0;
		return true;
	}
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &fun = expr.Cast<BoundFunctionExpression>();
	if (fun.function.GetStability() != FunctionStability::CONSISTENT) {
		return false;
	}
	if (fun.function.GetNullHandling() != FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		return false;
	}
	// `+` is commutative; per-arg metadata can't express "monotonic in either arg"
	if (fun.function.name == "+" && fun.children.size() == 2) {
		const bool lhs_foldable = fun.children[0]->IsFoldable();
		const bool rhs_foldable = fun.children[1]->IsFoldable();
		if (lhs_foldable != rhs_foldable) {
			step.col_arg = lhs_foldable ? 1 : 0;
			return true;
		}
	}
	const auto &monotonicity = fun.function.GetMonotonicity();
	if (!monotonicity.HasMonotonicArg()) {
		return false;
	}
	// Tier-1 (allow_finite_only=false): refuse functions that may NULL on infinity
	if (monotonicity.requires_finite_input && !allow_finite_only) {
		return false;
	}
	idx_t non_foldable = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < fun.children.size(); i++) {
		if (fun.children[i]->IsFoldable()) {
			continue;
		}
		if (non_foldable != DConstants::INVALID_INDEX) {
			return false; // more than one non-foldable arg, e.g. col1 - col2
		}
		non_foldable = i;
	}
	if (non_foldable == DConstants::INVALID_INDEX) {
		return false;
	}
	switch (monotonicity.GetOutputOrderForArg(non_foldable)) {
	case FunctionOutputOrder::MATCHES_INPUT_ORDER:
		break;
	case FunctionOutputOrder::INVERTS_INPUT_ORDER:
		step.inverts = true;
		break;
	case FunctionOutputOrder::UNSPECIFIED:
		return false;
	}
	step.col_arg = non_foldable;
	return true;
}

} // namespace duckdb
