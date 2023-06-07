An implementation of exactly evaluated floating point predicates for arbitrary expressions composed of additions and multiplications

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
There are 3 primary eval_methods:
* `correct_eval`: Returns an `std::optional` after evaluating with the specified floating point type if the result is guaranteed to have the correct sign based on relative error analysis. This computes relative error filters at compile-time, and if the maximum relative error ever exceeds 1, it returns nullopt. In the testing benchmarks this is about 3x slower than a basic floating point evaluation.
* `adaptive_eval`: Returns a value with the correct sign, but not necessarily a good representative of the exact result. Currently very slow, but potentially much faster than exact evaluation.
* `exact_eval`: Computes the result exactly, and then rounds it to the specified output format. The slowest method after adaptive_eval is improved

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
Multiplication computes the exact product of every pair in the two series being multiplied and creates a new series from the result
For example, (30.0 + 2.0) * (10.0 + 0.5) = (30.0 * 10.0 + 30.0 * 0.5 + 2.0 * 10.0 + 2.0 * 0.5),
which might expand to (300.0 + 10.0 + 5.0 + 20.0 + 1.0)
Division can't currently be computed exactly; but todo in the future, it will be handled by rewriting the expression in terms of multiplications.

This method is only performant for small expressions; for larger expressions, implementations using arbitrary precision floats like MPFR will be faster due to asymptotic complexity of multiplication
