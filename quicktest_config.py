from tests.test_autodiff import *
from tests.test_scalar import *

from minitorch.autodiff import *
from minitorch.scalar import *
from minitorch.scalar_functions import *

var = minitorch.Scalar(0.0, ScalarHistory())
constant = minitorch.Scalar(
    0.0, ScalarHistory(Function1, ctx=Context(), inputs=[var, var])
)
back = constant.chain_rule(5)
