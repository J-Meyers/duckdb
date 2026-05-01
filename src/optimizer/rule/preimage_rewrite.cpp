#include "duckdb/optimizer/rule/preimage_rewrite.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/function/arg_properties.hpp"
#include "duckdb/optimizer/expression_rewriter.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

namespace {

unique_ptr<Expression> MakeCmp(ExpressionType cmp, unique_ptr<Expression> col_expr, Value bound) {
	auto rhs = make_uniq<BoundConstantExpression>(std::move(bound));
	return make_uniq<BoundComparisonExpression>(cmp, std::move(col_expr), std::move(rhs));
}

unique_ptr<Expression> MakeBetween(unique_ptr<Expression> col_expr, Value lo, Value hi, bool lower_inclusive,
                                   bool upper_inclusive) {
	auto lower_const = make_uniq<BoundConstantExpression>(std::move(lo));
	auto upper_const = make_uniq<BoundConstantExpression>(std::move(hi));
	return make_uniq<BoundBetweenExpression>(std::move(col_expr), std::move(lower_const), std::move(upper_const),
	                                         lower_inclusive, upper_inclusive);
}

//! Identify the single column-bearing arg of `fun_expr` and return its properties.
//! Gates: CONSISTENT stability, default null handling, has-arg-props, known
//! monotonicity (not CONSTANT), preimage callback present, finite-input not required.
bool TryGetMonotonicArg(BoundFunctionExpression &fun_expr, idx_t &col_arg, const ArgProperties *&props_out) {
	if (fun_expr.function.GetStability() != FunctionStability::CONSISTENT) {
		return false;
	}
	if (fun_expr.function.GetNullHandling() != FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		return false;
	}
	if (!fun_expr.function.HasArgProperties()) {
		return false;
	}
	col_arg = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < fun_expr.children.size(); i++) {
		if (fun_expr.children[i]->IsFoldable()) {
			continue;
		}
		if (col_arg != DConstants::INVALID_INDEX) {
			return false;
		}
		col_arg = i;
	}
	if (col_arg == DConstants::INVALID_INDEX) {
		return false;
	}
	const auto &props = fun_expr.function.GetArgProperties(col_arg);
	if (!IsKnownMonotonic(props.monotonicity) || props.monotonicity == Monotonicity::CONSTANT) {
		return false;
	}
	if (props.preimage == nullptr) {
		return false;
	}
	if (props.requires_finite_input) {
		return false;
	}
	props_out = &props;
	return true;
}

struct PreimageInterval {
	Value lo;
	Value hi;
};

bool TryComputePreimage(BoundFunctionExpression &fun_expr, idx_t col_arg, const ArgProperties &props,
                        const Value &output_point, PreimageInterval &out) {
	if (output_point.IsNull()) {
		return false;
	}
	auto intervals = props.preimage(fun_expr, col_arg, output_point);
	// Multi-branch (e.g. abs, sqrt) bails for now.
	if (intervals.size() != 1) {
		return false;
	}
	out.lo = std::move(intervals[0].lo);
	out.hi = std::move(intervals[0].hi);
	if (out.lo.IsNull() || out.hi.IsNull()) {
		return false;
	}
	return true;
}

ExpressionType AdjustForMonotonicity(ExpressionType cmp_type, Monotonicity m) {
	if (IsMonotonicDecreasing(m)) {
		return FlipComparisonExpression(cmp_type);
	}
	return cmp_type;
}

unique_ptr<Expression> BuildPreimageComparison(ExpressionType cmp_type, unique_ptr<Expression> col_expr,
                                               const PreimageInterval &iv) {
	switch (cmp_type) {
	case ExpressionType::COMPARE_EQUAL:
		return MakeBetween(std::move(col_expr), iv.lo, iv.hi, true, false);
	case ExpressionType::COMPARE_NOTEQUAL: {
		auto col_copy = col_expr->Copy();
		auto disj = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_OR);
		disj->children.push_back(MakeCmp(ExpressionType::COMPARE_LESSTHAN, std::move(col_expr), iv.lo));
		disj->children.push_back(MakeCmp(ExpressionType::COMPARE_GREATERTHANOREQUALTO, std::move(col_copy), iv.hi));
		return std::move(disj);
	}
	case ExpressionType::COMPARE_GREATERTHAN:
		return MakeCmp(ExpressionType::COMPARE_GREATERTHANOREQUALTO, std::move(col_expr), iv.hi);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return MakeCmp(ExpressionType::COMPARE_GREATERTHANOREQUALTO, std::move(col_expr), iv.lo);
	case ExpressionType::COMPARE_LESSTHAN:
		return MakeCmp(ExpressionType::COMPARE_LESSTHAN, std::move(col_expr), iv.lo);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return MakeCmp(ExpressionType::COMPARE_LESSTHAN, std::move(col_expr), iv.hi);
	default:
		return nullptr;
	}
}

} // namespace

PreimageRewriteRule::PreimageRewriteRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	auto op = make_uniq<ComparisonExpressionMatcher>();
	op->matchers.push_back(make_uniq<ConstantExpressionMatcher>());
	auto fun_matcher = make_uniq<FunctionExpressionMatcher>();
	fun_matcher->policy = SetMatcher::Policy::SOME;
	op->matchers.push_back(std::move(fun_matcher));
	op->policy = SetMatcher::Policy::UNORDERED;
	root = std::move(op);
}

unique_ptr<Expression> PreimageRewriteRule::Apply(LogicalOperator &op, vector<reference<Expression>> &bindings,
                                                  bool &changes_made, bool is_root) {
	auto &comparison = bindings[0].get().Cast<BoundComparisonExpression>();
	// UNORDERED [Constant, Function]: bindings[1] = constant, bindings[2] = function.
	auto &constant_expr = bindings[1].get().Cast<BoundConstantExpression>();
	auto &fun_expr = bindings[2].get().Cast<BoundFunctionExpression>();
	const bool const_on_left = comparison.left.get() == &constant_expr;

	idx_t col_arg;
	const ArgProperties *props_ptr;
	if (!TryGetMonotonicArg(fun_expr, col_arg, props_ptr)) {
		return nullptr;
	}
	const auto &props = *props_ptr;

	PreimageInterval iv;
	if (!TryComputePreimage(fun_expr, col_arg, props, constant_expr.value, iv)) {
		return nullptr;
	}

	auto cmp_type = comparison.GetExpressionType();
	if (const_on_left) {
		cmp_type = FlipComparisonExpression(cmp_type);
	}
	cmp_type = AdjustForMonotonicity(cmp_type, props.monotonicity);

	auto col_expr = std::move(fun_expr.children[col_arg]);
	auto rewritten = BuildPreimageComparison(cmp_type, std::move(col_expr), iv);
	if (!rewritten) {
		// DISTINCT_FROM / NOT_DISTINCT_FROM — restore the original child.
		fun_expr.children[col_arg] = std::move(col_expr);
		return nullptr;
	}
	changes_made = true;
	return rewritten;
}

PreimageBetweenRewriteRule::PreimageBetweenRewriteRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	auto op = make_uniq<BetweenExpressionMatcher>();
	auto fun_matcher = make_uniq<FunctionExpressionMatcher>();
	fun_matcher->policy = SetMatcher::Policy::SOME;
	op->matchers.push_back(std::move(fun_matcher));                 // input
	op->matchers.push_back(make_uniq<ConstantExpressionMatcher>()); // lower
	op->matchers.push_back(make_uniq<ConstantExpressionMatcher>()); // upper
	root = std::move(op);
}

unique_ptr<Expression> PreimageBetweenRewriteRule::Apply(LogicalOperator &op, vector<reference<Expression>> &bindings,
                                                         bool &changes_made, bool is_root) {
	auto &between = bindings[0].get().Cast<BoundBetweenExpression>();
	auto &fun_expr = bindings[1].get().Cast<BoundFunctionExpression>();
	auto &lower_const = bindings[2].get().Cast<BoundConstantExpression>();
	auto &upper_const = bindings[3].get().Cast<BoundConstantExpression>();

	idx_t col_arg;
	const ArgProperties *props_ptr;
	if (!TryGetMonotonicArg(fun_expr, col_arg, props_ptr)) {
		return nullptr;
	}
	const auto &props = *props_ptr;

	PreimageInterval lower_iv;
	PreimageInterval upper_iv;
	if (!TryComputePreimage(fun_expr, col_arg, props, lower_const.value, lower_iv)) {
		return nullptr;
	}
	if (!TryComputePreimage(fun_expr, col_arg, props, upper_const.value, upper_iv)) {
		return nullptr;
	}

	// For monotonic-decreasing f, input bounds swap (input upper <- output lower etc.)
	Value out_lo, out_hi;
	const bool decreasing = IsMonotonicDecreasing(props.monotonicity);
	if (!decreasing) {
		out_lo = between.lower_inclusive ? lower_iv.lo : lower_iv.hi;
		out_hi = between.upper_inclusive ? upper_iv.hi : upper_iv.lo;
	} else {
		out_lo = between.upper_inclusive ? upper_iv.lo : upper_iv.hi;
		out_hi = between.lower_inclusive ? lower_iv.hi : lower_iv.lo;
	}

	auto col_expr = std::move(fun_expr.children[col_arg]);
	changes_made = true;
	return MakeBetween(std::move(col_expr), std::move(out_lo), std::move(out_hi), true, false);
}

} // namespace duckdb
