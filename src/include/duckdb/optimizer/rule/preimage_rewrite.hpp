//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/rule/preimage_rewrite.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/rule.hpp"

namespace duckdb {

//! Rewrites `f(col) <cmp> literal` into a predicate on `col` directly when `f`'s
//! ArgProperties on the column-bearing arg supplies a preimage callback.
//!
//! For a STRICTLY_INCREASING f with preimage(c) = [lo, hi):
//!   f(x) =  c   ->   x BETWEEN lo AND hi (lower-incl, upper-excl)
//!   f(x) <> c   ->   x <  lo OR  x >= hi
//!   f(x) >  c   ->   x >= hi
//!   f(x) >= c   ->   x >= lo
//!   f(x) <  c   ->   x <  lo
//!   f(x) <= c   ->   x <  hi
//! For STRICTLY_DECREASING the inequality directions mirror.
class PreimageRewriteRule : public Rule {
public:
	explicit PreimageRewriteRule(ExpressionRewriter &rewriter);

	unique_ptr<Expression> Apply(LogicalOperator &op, vector<reference<Expression>> &bindings, bool &changes_made,
	                             bool is_root) override;
};

//! Same rewrite for `f(col) BETWEEN lo AND hi`. Output BETWEEN is always
//! [lower_inclusive=true, upper_inclusive=false) in canonical form.
class PreimageBetweenRewriteRule : public Rule {
public:
	explicit PreimageBetweenRewriteRule(ExpressionRewriter &rewriter);

	unique_ptr<Expression> Apply(LogicalOperator &op, vector<reference<Expression>> &bindings, bool &changes_made,
	                             bool is_root) override;
};

} // namespace duckdb
