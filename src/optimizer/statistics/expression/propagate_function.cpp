#include "duckdb/optimizer/statistics_propagator.hpp"

#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/arg_properties.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"

namespace duckdb {

static bool TryEvaluateAtConstants(ClientContext &context, const BoundFunctionExpression &func,
                                   const vector<Value> &arg_values, Value &result) {
	vector<unique_ptr<Expression>> children;
	children.reserve(arg_values.size());
	for (auto &v : arg_values) {
		children.push_back(make_uniq<BoundConstantExpression>(v));
	}
	auto bind_info_clone = func.bind_info ? func.bind_info->Copy() : nullptr;
	BoundFunctionExpression clone(func.GetReturnType(), func.function, std::move(children), std::move(bind_info_clone),
	                              func.is_operator);
	return ExpressionExecutor::TryEvaluateScalar(context, clone, result);
}

//! Derive output stats for a BoundFunctionExpression by evaluating it at the lo/hi corner of
//! each arg's input range, using the per-arg monotonicity declaration to flip decreasing args.
//! Returns nullptr when any precondition fails (unknown monotonicity, missing min/max, evaluation
//! throws, etc.).
static unique_ptr<BaseStatistics> TryPropagateMonotoneBounds(ClientContext &context, BoundFunctionExpression &func,
                                                             const vector<BaseStatistics> &child_stats) {
	if (!func.function.HasArgProperties() || func.children.empty()) {
		return nullptr;
	}
	if (func.function.GetStability() != FunctionStability::CONSISTENT) {
		return nullptr;
	}
	if (func.function.GetNullHandling() != FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		return nullptr;
	}
	// only NumericStats carry the min/max needed for corner evaluation
	if (BaseStatistics::GetStatsType(func.GetReturnType()) != StatisticsType::NUMERIC_STATS ||
	    func.GetReturnType().InternalType() == PhysicalType::BOOL) {
		return nullptr;
	}

	vector<Value> lo_args(func.children.size());
	vector<Value> hi_args(func.children.size());
	bool any_input_can_have_null = false;

	for (idx_t i = 0; i < func.children.size(); i++) {
		auto &child = *func.children[i];
		auto &cs = child_stats[i];
		auto &props = func.function.GetArgProperties(i);

		if (child.IsFoldable()) {
			Value v;
			if (!ExpressionExecutor::TryEvaluateScalar(context, child, v) || v.IsNull()) {
				return nullptr;
			}
			lo_args[i] = v;
			hi_args[i] = std::move(v);
			continue;
		}

		auto m = props.monotonicity;
		if (cs.CanHaveNull()) {
			any_input_can_have_null = true;
		}
		if (cs.GetStatsType() != StatisticsType::NUMERIC_STATS || !NumericStats::HasMinMax(cs)) {
			return nullptr;
		}
		Value lo = NumericStats::Min(cs);
		Value hi = NumericStats::Max(cs);

		if (props.refine_monotonicity) {
			m = props.refine_monotonicity(func, i, lo, hi);
		}
		if (!IsKnownMonotonic(m)) {
			return nullptr;
		}
		// requires_finite_input args (e.g. year(date), date_diff) NULL on +/-infinity inputs.
		// We don't pre-check; the corner evaluator returns NULL or throws on those, both caught below.

		if (m == Monotonicity::CONSTANT) {
			lo_args[i] = lo;
			hi_args[i] = std::move(lo);
			continue;
		}
		if (IsMonotonicIncreasing(m)) {
			lo_args[i] = std::move(lo);
			hi_args[i] = std::move(hi);
		} else {
			D_ASSERT(IsMonotonicDecreasing(m));
			lo_args[i] = std::move(hi);
			hi_args[i] = std::move(lo);
		}
	}

	Value out_lo, out_hi;
	if (!TryEvaluateAtConstants(context, func, lo_args, out_lo) ||
	    !TryEvaluateAtConstants(context, func, hi_args, out_hi)) {
		return nullptr;
	}
	if (out_lo.IsNull() || out_hi.IsNull()) {
		return nullptr;
	}
	if (out_hi < out_lo) {
		// a refiner misclassified at a boundary; defensive normalize
		std::swap(out_lo, out_hi);
	}

	auto result = NumericStats::CreateEmpty(func.GetReturnType());
	NumericStats::SetMin(result, out_lo);
	NumericStats::SetMax(result, out_hi);

	if (func.function.HasCodomainBounds()) {
		auto &cap = func.function.GetCodomainBounds();
		if (!cap.lo.IsNull() && NumericStats::Min(result) < cap.lo) {
			NumericStats::SetMin(result, cap.lo);
		}
	}

	result.Set(StatsInfo::CAN_HAVE_VALID_VALUES);
	if (any_input_can_have_null) {
		result.Set(StatsInfo::CAN_HAVE_NULL_VALUES);
	}
	return result.ToUnique();
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundFunctionExpression &func,
                                                                     unique_ptr<Expression> &expr_ptr) {
	vector<BaseStatistics> stats;
	stats.reserve(func.children.size());
	for (idx_t i = 0; i < func.children.size(); i++) {
		auto stat = PropagateExpression(func.children[i]);
		if (!stat) {
			stats.push_back(BaseStatistics::CreateUnknown(func.children[i]->GetReturnType()));
		} else {
			stats.push_back(stat->Copy());
		}
	}
	if (func.function.HasStatisticsCallback()) {
		FunctionStatisticsInput input(func, func.bind_info.get(), stats, &expr_ptr);
		return func.function.GetStatisticsCallback()(context, input);
	}
	return TryPropagateMonotoneBounds(context, func, stats);
}

} // namespace duckdb
