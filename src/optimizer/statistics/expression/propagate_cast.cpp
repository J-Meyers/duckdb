#include "duckdb/optimizer/statistics_propagator.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/storage/statistics/struct_stats.hpp"
#include "duckdb/storage/statistics/variant_stats.hpp"

namespace duckdb {

static unique_ptr<BaseStatistics> StatisticsOperationsNumericNumericCast(const BaseStatistics &input,
                                                                         const LogicalType &target) {
	// Bail out if the stats are not numeric
	if (input.GetStatsType() != StatisticsType::NUMERIC_STATS) {
		return nullptr;
	}
	if (!NumericStats::HasMinMax(input)) {
		return nullptr;
	}
	Value min = NumericStats::Min(input);
	Value max = NumericStats::Max(input);
	if (!min.DefaultTryCastAs(target) || !max.DefaultTryCastAs(target)) {
		// overflow in cast: bailout
		return nullptr;
	}
	auto result = NumericStats::CreateEmpty(target);
	result.CopyBase(input);
	NumericStats::SetMin(result, min);
	NumericStats::SetMax(result, max);
	return result.ToUnique();
}

bool StatisticsPropagator::CanPropagateCast(const LogicalType &source, const LogicalType &target) {
	if (source == target) {
		return true;
	}
	// we can only propagate numeric -> numeric
	switch (source.InternalType()) {
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		break;
	default:
		return false;
	}
	switch (target.InternalType()) {
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		break;
	default:
		return false;
	}
	// for time/timestamps/dates - there are various limitations on what we can propagate
	//	Downcasting timestamps to times is not a truncation operation
	switch (target.id()) {
	case LogicalTypeId::TIME: {
		switch (source.id()) {
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_SEC:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::TIMESTAMP_TZ:
			return false;
		default:
			break;
		}
		break;
	}
	// FIXME: perform actual stats propagation for these casts
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ: {
		const bool to_timestamp = target.id() == LogicalTypeId::TIMESTAMP;
		const bool to_timestamp_tz = target.id() == LogicalTypeId::TIMESTAMP_TZ;
		//  Casting to timestamp[_tz] (us) from a different unit can not re-use stats
		switch (source.id()) {
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP_SEC:
			return false;
		case LogicalTypeId::TIMESTAMP: {
			if (to_timestamp_tz) {
				// Both use INT64 physical type, but should not be treated equal
				return false;
			}
			break;
		}
		case LogicalTypeId::TIMESTAMP_TZ: {
			if (to_timestamp) {
				// Both use INT64 physical type, but should not be treated equal
				return false;
			}
			break;
		}
		default:
			break;
		}
		break;
	}
	case LogicalTypeId::TIMESTAMP_NS: {
		// Same as above ^
		switch (source.id()) {
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_TZ:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP_SEC:
			return false;
		default:
			break;
		}
		break;
	}
	case LogicalTypeId::TIMESTAMP_MS: {
		// Same as above ^
		switch (source.id()) {
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_TZ:
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::TIMESTAMP_SEC:
			return false;
		default:
			break;
		}
		break;
	}
	case LogicalTypeId::TIMESTAMP_SEC: {
		// Same as above ^
		switch (source.id()) {
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_TZ:
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::TIMESTAMP_MS:
			return false;
		default:
			break;
		}
		break;
	}
	case LogicalTypeId::TIME_TZ: {
		// Casts to TIMETZ from TIME or TIMESTAMPTZ are session-TimeZone dependent
		// (the ICU extension overrides them at execution time) but Value::DefaultTryCastAs
		// uses the static, UTC-only operator, so propagated min/max would diverge from
		// runtime values. See issue #22235.
		switch (source.id()) {
		case LogicalTypeId::TIME:
		case LogicalTypeId::TIMESTAMP_TZ:
			return false;
		default:
			break;
		}
		break;
	}
	default:
		break;
	}
	// we can propagate!
	return true;
}

static unique_ptr<BaseStatistics> StatisticsPropagateVariant(const BaseStatistics &input, const LogicalType &target) {
	if (target.IsNested() || target.id() == LogicalTypeId::VARIANT) {
		// only try this for non-nested
		return nullptr;
	}
	if (!VariantStats::IsShredded(input)) {
		// not shredded
		return nullptr;
	}
	auto structured_type = VariantStats::GetShreddedStructuredType(input);
	auto &shredded_stats = VariantStats::GetShreddedStats(input);
	if (!VariantShreddedStats::IsFullyShredded(shredded_stats)) {
		// this field might be partially shredded - skip stats propagation
		return nullptr;
	}
	// extract the typed stats
	auto &typed_stats = VariantStats::GetTypedStats(shredded_stats);
	if (structured_type == target) {
		// type matches - return stats directly
		return typed_stats.ToUnique();
	}
	// typed stats don't match - try to cast
	return StatisticsPropagator::TryPropagateCast(typed_stats, structured_type, target);
}

unique_ptr<BaseStatistics> StatisticsPropagator::TryPropagateCast(const BaseStatistics &stats,
                                                                  const LogicalType &source,
                                                                  const LogicalType &target) {
	if (source.id() == LogicalTypeId::VARIANT) {
		return StatisticsPropagateVariant(stats, target);
	}
	if (!CanPropagateCast(source, target)) {
		return nullptr;
	}
	return StatisticsOperationsNumericNumericCast(stats, target);
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundCastExpression &cast,
                                                                     unique_ptr<Expression> &expr_ptr) {
	auto child_stats = PropagateExpression(cast.child);
	if (!child_stats) {
		return nullptr;
	}
	auto &source = cast.child->GetReturnType();
	auto &target = cast.GetReturnType();

	// VARIANT source goes through the shredded-stats path (still uses the type-pair allowlist).
	if (source.id() == LogicalTypeId::VARIANT) {
		auto result = StatisticsPropagateVariant(*child_stats, target);
		if (cast.try_cast && result) {
			result->Set(StatsInfo::CAN_HAVE_NULL_VALUES);
		}
		return result;
	}

	// Identity casts (source == target) preserve stats directly. The cast registry returns
	// `NopCast` here without arg_properties, so handle this before reading metadata.
	if (source == target) {
		auto result = child_stats->ToUnique();
		if (cast.try_cast && result) {
			result->Set(StatsInfo::CAN_HAVE_NULL_VALUES);
		}
		return result;
	}

	// Read the bound cast's per-arg metadata. For monotonic casts we map child min/max through
	// the static Value cast to derive output stats. The metadata is set per-bound-cast at the
	// `*CastSwitch` registration site, so ICU overrides (which depend on session zone / DST)
	// land here as UNKNOWN and the propagation correctly bails.
	auto &props = cast.bound_cast.GetArgProperties();
	if (!IsKnownMonotonic(props.monotonicity)) {
		return nullptr;
	}
	if (BaseStatistics::GetStatsType(target) != StatisticsType::NUMERIC_STATS) {
		// String / nested / interval output stats can't be derived this way.
		return nullptr;
	}
	auto result_stats = StatisticsOperationsNumericNumericCast(*child_stats, target);
	if (result_stats && IsMonotonicDecreasing(props.monotonicity)) {
		// Decreasing cast: cast(min) is the new max, cast(max) is the new min.
		Value swapped_min = NumericStats::Max(*result_stats);
		Value swapped_max = NumericStats::Min(*result_stats);
		NumericStats::SetMin(*result_stats, swapped_min);
		NumericStats::SetMax(*result_stats, swapped_max);
	}
	if (cast.try_cast && result_stats) {
		result_stats->Set(StatsInfo::CAN_HAVE_NULL_VALUES);
	}
	return result_stats;
}

} // namespace duckdb
