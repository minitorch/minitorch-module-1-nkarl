from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from minitorch import operators as ops
from minitorch.autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# NOTE: might not need to implement for operators because python provides unpacking
def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """
        calls a node's backward to produces a tensor of partial derivatives.
        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inputs: float) -> float:
        """
        calls a node's forward on the node to evaluate.
        """
        return cls.forward(ctx, *inputs)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        """
        computes and creates a new Scalar object.
            1. process scalar-like data
            2. compute using forward
                - save the input into context
                - compute and return raw float
            3. create new scalar object from result from #2.
        """
        raw_vals = []
        scalars = []

        # 1. prep and process
        #   - filter *Scalar-like* data into two lists, raw values and MiniTorch Scalars.
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create empty context.
        ctx = Context(False)

        # 2. save context and produce result
        c = cls._forward(ctx, *raw_vals)  # NOTE: operator's eval result of type FLOAT
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # BUG: missing local backward
        g = sum(cls._backward(ctx, 1.0))

        # 3. Create a new variable from the result with a new history.
        trace = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        new = minitorch.scalar.Scalar(c, trace)
        new.derivative = g

        return new


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        x, y = ops.id(x), ops.id(y)
        ctx.save_for_backward(x, y)
        return ops.add(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return (d_output, d_output)


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        x = ops.id(x)
        ctx.save_for_backward(x)
        return ops.log(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        return ops.log_back(x, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        x, y = ops.id(x), ops.id(y)
        ctx.save_for_backward(x, y)
        return ops.mul(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        grad_y, grad_x = ctx.saved_values
        # order of the tuple doesn't actually matter, because it gets folded
        return (d_output * grad_y, d_output * grad_x)


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        x = ops.id(x)
        ctx.save_for_backward(x)
        return ops.inv(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        grad_x = ops.neg(ops.mul(d_output, x))
        return grad_x


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        x = ops.id(x)
        ctx.save_for_backward(x)
        return ops.neg(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        grad_x = ops.mul(d_output, x)
        return grad_x


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        ctx.save_for_backward(x)
        return ops.sigmoid(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        if x >= 0.0:
            return d_output * 1.0
        else:
            return d_output * (1.0 + ops.exp(x))


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        x = ops.id(x)
        ctx.save_for_backward(x)
        return ops.relu(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        grad_x = ops.relu_back(x, d_output)
        return grad_x


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        ctx.save_for_backward(x)
        return ops.exp(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        grad_x = d_output * ops.exp(x)
        return grad_x


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        x, y = ops.id(x), ops.id(y)
        ctx.save_for_backward(x, y)
        return ops.lt(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # x, y = ctx.saved_values
        return d_output, d_output


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        x, y = ops.id(x), ops.id(y)
        ctx.save_for_backward(x, y)
        return ops.eq(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # x, y = ctx.saved_values
        return d_output, d_output
