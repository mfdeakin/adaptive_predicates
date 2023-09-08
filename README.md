An implementation of exactly evaluated floating point predicates for arbitrary expressions composed of additions and multiplications.

This provides an expression template system for representing arbitrary expressions at compile-time,
and eval functions for evaluating the expressions with various guarantees.

Example:
```
auto expr = (mult_expr(5, 8.0) + 12.0 - 5.6) * 0.23;
std::optional<float> evaluatable = correct_eval<float>(expr * 32.2);
if(!evaluatable) {
    float exact_sign = adaptive_eval<float>(expr * 32.0);
    float rounded_exact_val = exact_eval<float>(expr * 32.0);
}
```
There are 5 primary eval_methods:
* `correct_eval`: Returns an `std::optional` after evaluating with the specified floating point type if the result is guaranteed to have the correct sign based on relative error analysis. This uses `eval_checked_fast` and `eval_with_err` internally; if both of those fail to give a result with the correct sign, it returns `std::nullopt`. If you need to run on vector types, use `eval_checked_fast` or `eval_with_err` directly instead.
* `eval_checked_fast`: Returns a pair with the approximate result as the first parameter, and a bool indicating whether relative error analysis is useful on the result. If the result isn't guaranteed to have the correct sign based on relative error analysis, then the first return is NaN. This works with `vector_type`. Note that this is not as discerning as `eval_with_err`, and it can only function with at most one addition of opposite signs (or subtraction of the same signs) above the 2nd level of the expression tree, though when it works it is much faster with expressions with only a few additions/subtractions above the 2nd level of the expression tree.
* `eval_with_err`: Returns a pair containing the approximate result and the estimated error if the result is guaranteed to have the correct sign based on interval arithmetic. If not, then it returns NaN. This works with vector types with an associated `select` method.
* `adaptive_eval`: Returns a value with the correct sign, but not necessarily a good representative of the exact result. For latency sensitive applications, use this. This runs `eval_checked_fast` before attempting to perform adaptive evaluation, and if that fails, uses interval arithmetic to ensure that no sign errors can occur. After that, it uses a simple compile-time latency analysis to try to choose the sub-tree to evaluate. This does not support vector types, and adaptive evaluation that ends up exactly evaluating the entire expression will be more expensive than `exact_eval`.
* `exact_eval`: Computes the result exactly, and then rounds it to the specified output format. This is consistently very slow, however, unlike adaptive_eval, it supports running on SIMD types that satisfy the `vector_type` concept. For high throughput applications, use this on expressions after determining they need exact evaluation.

**Note:** To run efficiently on a GPU, use the provided `GPUVec` type as the `eval_type` parameter to `exact_eval`

The `arith_expr` class represents the expression template as a binary tree; it takes 3 template parameters:
* `Op`: A functor representing the arithmetic expression to be applied; generally one of `std::plus`, `std::minus`, or `std::multiplies`
* `LHS`: The expression (or number) on the left hand side of the operator
* `RHS`: The expression (or number) on the right hand side of the operator
and provides 2 methods:
* `lhs`: Returns the expression (or number) on the left hand side of the operator
* `rhs`: Returns the expression (or number) on the right hand side of the operator

This computes exactly by representing intermediate results as a finite series
For example, 435.75 might be represented as (430.0 + 5.0 + 0.75), or as (195.0 - 200.0 + 195.0 + 200.75 - 55.0)
Then, results are computed exactly by appending the necessary values to the series
Addition and subtraction are straight-forward, they just append the operand to the series
Multiplication computes the exact product of every pair in the two series being multiplied and creates a new series from the result.
For example, (30.0 + 2.0) * (10.0 + 0.5) = (30.0 * 10.0 + 30.0 * 0.5 + 2.0 * 10.0 + 2.0 * 0.5),
which might expand to (300.0 + 10.0 + 5.0 + 20.0 + 1.0).
Division can't currently be computed exactly; but todo in the future, it will be handled by rewriting the expression in terms of multiplications.

There are three concepts that are important to this library:
* `arith_number`: Any data-type that has `+`, `-`, `*`, operators and functions `abs` and `mul_sub`.
`abs` is the regular absolute value function, and `mul_sub` computes `a * b - c` rounded to `1/2 epsilon` for your data-type.
`mul_sub` is typically implemented with `fma` as `fma(a, b, -c)`
* `scalar_type`: Any `arith_number` that does not have an `[]` operator.
* `vector_type`: Any `arith_number` that also has an `[]` operator and a `select` method.
`select` takes 3 parameters: the first a parameter `p` with the type returned by `v1 > v2` for `vector_type`s `v1` and `v2`, and 2 `vector_type` parameters, and returns a `vector_type` parameter containing elements from `v1` when the corresponding element in `p` is true or the element from `v2` when false.
