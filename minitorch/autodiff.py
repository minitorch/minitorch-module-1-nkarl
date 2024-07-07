from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
import graphviz

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an _approximation_ to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f       : an arbitrary *reducer* function; *folds* $n$ scalar args to one value
        *vals   : a tensor of $n$ float values $x_0 \ldots x_{n-1}$
        arg     : the $i$ of the arg to compute the derivative
        epsilon : a small stepsize constant $h$

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    right, left = [*vals], [*vals]
    right[arg] += epsilon  # t[x+h]
    left[arg] -= epsilon  # t[x-h]
    return (f(*right) - f(*left)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...

    @property
    def unique_id(self) -> int: ...

    def is_leaf(self) -> bool: ...

    def is_constant(self) -> bool: ...

    @property
    def parents(self) -> Iterable["Variable"]: ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]: ...


# init graph builder
dot = graphviz.Digraph(
    graph_attr={"fontsize": "32", "penwidth": "2"},
    node_attr={"fontsize": "32", "penwidth": "2"},
    edge_attr={"fontsize": "32", "penwidth": "2"},
)


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    map: List[Variable] = []
    visited = set()

    dot.node(
        f"{variable.unique_id}",
        f"{variable.unique_id},\ndata={variable.data},\ngrad={variable.derivative}",
    )

    def map_from(var: Variable) -> None:
        if var.unique_id in visited:  # NOTE: DEP ATTRIBUTE
            return
        dot.node(
            name=f"{var.unique_id}",
            label=f"{var.unique_id},\ndata={var.data},\ngrad={var.derivative}",
        )
        # if var is not variable:
        # dot.edge(f"{variable.unique_id}", f"{var.unique_id}")
        if not var.is_leaf():  # NOTE: DEP CALL
            for each in var.history.inputs:  # NOTE: DEP ATTRIBUTE
                if not each.is_constant():
                    map_from(each)  # RECURSIVE CALL
                dot.node(
                    name=f"{each.unique_id}",
                    label=f"{each.unique_id},\ndata={each.data},\ngrad={each.derivative}",
                )
                dot.edge(f"{var.unique_id}", f"{each.unique_id}")

        visited.add(var.unique_id)
        map.insert(0, var)

    map_from(variable)
    return map


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        var: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    # TODO: Implement for Task 1.4.

    graph = topological_sort(variable)
    print("BEOFRE-------------------------------------------------")
    # for each in graph:
        # printinfo(each)
    d_dict = {}
    d_dict[variable.unique_id] = deriv

    for var in graph:
        d = d_dict.get(var.unique_id, deriv)
        var.derivative = d
        # NOTE: if not leaf, collect the sum(partial derivatives) and pass it on to its inputs
        #   NEED TO WRITE SHIT DOWN BRO
        if not var.is_leaf():
            h = var.history
            # print(h)
            # out = h.last_fn.backward(h.ctx, h.inputs)  # BUG: problem? the second param should be a float
            out = h.last_fn.backward(h.ctx, d)
            d_dict[var.unique_id] = sum(out)

            dot.node(
                name=f"{var.unique_id}",
                label=f"INNER {var.unique_id}\nd={d},\ndata={var.data},\ngrad={var.derivative}",
            )
            # print(f"out={out}")
            # for v, d_part in out:
                # d_dict[v.unique_id] = d_dict.get(v.unique_id, deriv)
                # dot.node(
                    # name=f"{v.unique_id}",
                    # label=f"INNER {v.unique_id}\nd={d_part},\ndata={v.data},\ngrad={v.derivative}",
                # )
            # dot.edge(f"{v.unique_id}", f"{var.unique_id}")
        else:
            # BUG: only 1 leaf has grad, the rest do not
            d_dict[var.unique_id] = d_dict.get(var.unique_id, 0)
            var.name = "LEAF"
            var.accumulate_derivative(d_dict.get(var.unique_id, d))  # BUG: DEP CALL
            dot.node(
                name=f"{var.unique_id}",
                label=f"LEAF {var.unique_id}\ndata={var.data},\ngrad={var.derivative}",
                color="red",
                style="filled",
                fillcolor="lightsteelblue",
            )
        printinfo(var)

    dot.render("graph", format="svg", view=True)

    print()
    print("AFTER-------------------------------------------------")
    # print(f"graph ={graph}")
    # for each in graph:
        # printinfo(each)
    print(f"d_dict={d_dict}")

    variable.graph = graph if graph else ["Something wrong"]
    variable.d_dict = d_dict if d_dict else {"Something wrong": None}

    #
    # NOTE: the variable info must be filled in, including the accumulated derivative.


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values


def resize_graph(dot, size_per_element=0.25, min_size=20):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def printinfo(var):
    # info = get_info(var)
    print(f"id={var.unique_id}")
    print(f"name={var.name}")
    print(f"\tdata={var.data}")
    print(f"\tgrad={var.derivative}")
    print(f"\thistory.last_fn={var.history.last_fn}")
    print(f"\thistory.ctx={var.history.ctx}")
    print(f"\thistory.(x,y)={var.history.inputs}")
    # x, y = var.history.inputs
    # _printinfo(x)
    # _printinfo(y)


def get_info(var):
    return {
        "history": var.history,
        "derivative": var.derivative,
        "data": var.data,
        "unique_id": var.unique_id,
        "name": var.name,
    }
