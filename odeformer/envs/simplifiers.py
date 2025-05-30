# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.basic import preorder_traversal
from sympy.core.rules import Transform
from functools import partial
import torch
import concurrent.futures

from .generators import all_operators, math_constants, Node, NodeList
# from ..utils import timeout, MyTimeoutError

class InvalidPrefixExpression(BaseException):
    pass


import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError

def run_with_timeout(func, *args, timeout_sec=1, **kwargs):
    """
    Run a function with the given arguments, enforcing a timeout.
    Returns the function result, or the first argument (e.g., expr) on timeout.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            # Return the first argument (usually expr) on timeout
            return args[0] if args else None

class Simplifier(ABC):

    local_dict = {
        "n": sp.Symbol("n", real=True, nonzero=True, positive=True, integer=True),
        "e": sp.E,
        "pi": sp.pi,
        "euler_gamma": sp.EulerGamma,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "step": sp.Heaviside,
        "sign": sp.sign,
    }
    for d in range(10):
        k = "x_{}".format(d)
        local_dict[k] = sp.Symbol(k, real=True, integer=False)

    def __init__(self, generator):

        self.params = generator.params
        self.encoder = generator.equation_encoder

        for k in generator.variables:
            self.local_dict[k] = sp.Symbol(k, real=True, integer=False)

    def simplify_tree(self, tree, expand=False, resimplify=False, round=False, decimals=3):
        if hasattr(tree, "nodes"):
            return NodeList([self.simplify_tree(node, expand, resimplify, round, decimals) for node in tree.nodes])
        else:
            if tree is None:
                return tree
            expr = self.tree_to_sympy_expr(tree)
            if expand:
                expr = self.expand_expr(expr)
            if resimplify:
                expr = self.simplify_expr(expr)
            new_tree = self.sympy_expr_to_tree(expr)
            if new_tree is None:
                return tree
            else:
                return new_tree.nodes[0]
        
    @classmethod
    def readable_tree(cls, tree):
        if tree is None:
            return None
        tree_sympy = cls.tree_to_sympy_expr(tree, round=True)
        readable_tree = '  ,  '.join([str(tree) for tree in tree_sympy])
        return readable_tree

    @classmethod
    def tree_to_sympy_expr(cls, tree, round=True):
        if hasattr(tree, 'nodes'):
            return [cls.tree_to_sympy_expr(node, round=round) for node in tree.nodes]
        prefix = tree.prefix().split(",")
        sympy_compatible_infix = cls.prefix_to_sympy_compatible_infix(prefix)
        expr = parse_expr(
            sympy_compatible_infix, evaluate=True, local_dict=cls.local_dict
        )
        if round: expr = cls.round_expr(expr)
        return expr
    
    @classmethod
    def _prefix_to_sympy_compatible_infix(cls, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in all_operators:
            args = []
            l1 = expr[1:]
            for _ in range(all_operators[t]):
                i1, l1 = cls._prefix_to_sympy_compatible_infix(l1)
                args.append(i1)
            return cls.write_infix(t, args), l1
        else:  # leaf
            try:
                float(t)
                t = str(t)
            except ValueError:
                t = t
            return t, expr[1:]

    @classmethod
    def prefix_to_sympy_compatible_infix(cls, expr):
        """
        Convert prefix expressions to a format that SymPy can parse.
        """
        p, r = cls._prefix_to_sympy_compatible_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(
                f'Incorrect prefix expression "{expr}". "{r}" was not parsed.'
            )
        return f"({p})"
    
    @classmethod
    def round_expr(cls, expr, decimals=4, timeout_sec=1):
        def _round():
            return expr.xreplace(
                Transform(
                    lambda x: x.round(decimals), lambda x: isinstance(x, sp.Float)
                )
            )
        return run_with_timeout(_round, timeout_sec=timeout_sec)

    def expand_expr(self, expr, timeout_sec=1):
        def _expand():
            return sp.expand(expr)
        return run_with_timeout(_expand, timeout_sec=timeout_sec)

    def simplify_expr(self, expr, timeout_sec=1):
        def _simplify():
            return sp.simplify(expr)
        return run_with_timeout(_simplify, timeout_sec=timeout_sec)

    def tree_to_torch_module(self, tree, dtype=torch.float32):
        expr = self.tree_to_sympy_expr(tree)
        mod = self.expr_to_torch_module(expr, dtype)
        return mod

    def expr_to_numpy_fn(self, expr):
        def wrapper_fn(_expr, x, extra_local_dict={}):
            local_dict = {}
            for d in range(x.shape[1]):
                local_dict["x_{}".format(d)] = x[:, d]
            local_dict.update(extra_local_dict)
            variables_symbols = sp.symbols(
                " ".join(["x_{}".format(d) for d in range(x.shape[1])])
            )
            extra_symbols = list(extra_local_dict.keys())
            if len(extra_symbols) > 0:
                extra_symbols = sp.symbols(" ".join(extra_symbols))
            else:
                extra_symbols = ()
            np_fn = sp.lambdify(
                (*variables_symbols, *extra_symbols), _expr, modules="numpy"
            )
            return np_fn(**local_dict)

        return partial(wrapper_fn, expr)

    def tree_to_numpy_fn(self, tree):
        expr = self.tree_to_sympy_expr(tree)
        return self.expr_to_numpy_fn(expr)

    def sympy_expr_to_tree(self, expr):
        prefix = self.sympy_to_prefix(expr)
        return self.encoder.decode(prefix)

    def float_to_int_expr(self, expr):
        floats = expr.atoms(sp.Float)
        ints = [fl for fl in floats if int(fl) == fl]
        expr = expr.xreplace(dict(zip(ints, [int(i) for i in ints])))
        return expr

    @classmethod
    def write_infix(cls, token, args):
        """
        Infix representation.
    
        """
        if token == "add":
            return f"({args[0]})+({args[1]})"
        elif token == "sub":
            return f"({args[0]})-({args[1]})"
        elif token == "mul":
            return f"({args[0]})*({args[1]})"
        elif token == "div":
            return f"({args[0]})/({args[1]})"
        if token == "pow":
            return f"({args[0]})**({args[1]})"
        elif token == "idiv":
            return f"idiv({args[0]},{args[1]})"
        elif token == "mod":
            return f"({args[0]})%({args[1]})"
        elif token == "abs":
            return f"Abs({args[0]})"
        elif token == "id":
            return f"{args[0]}"
        elif token == "inv":
            return f"1/({args[0]})"
        elif token == "pow2":
            return f"({args[0]})**2"
        elif token == "pow3":
            return f"({args[0]})**3"
        elif token in all_operators:
            return f"{token}({args[0]})"
        else:
            return token
        raise InvalidPrefixExpression(
            f"Unknown token in prefix expression: {token}, with arguments {args}"
        )

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        # assert (op == 'add' or op == 'mul') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <= 2)

        # square root
        # if op == 'pow':
        #     if isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
        #         return ['sqrt'] + self.sympy_to_prefix(expr.args[0])
        #     elif str(expr.args[1])=='2':
        #         return ['sqr'] + self.sympy_to_prefix(expr.args[0])
        #     elif str(expr.args[1])=='-1':
        #         return ['inv'] + self.sympy_to_prefix(expr.args[0])
        #     elif str(expr.args[1])=='-2':
        #         return ['inv', 'sqr'] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]
        elif isinstance(expr, sp.Float):
            s = str(expr)
            return [s]
        elif isinstance(expr, sp.Rational):
            return ["mul", str(expr.p), "pow", str(expr.q), "-1"]
        elif expr == sp.EulerGamma:
            return ["euler_gamma"]
        elif expr == sp.E:
            return ["e"]
        elif expr == sp.pi:
            return ["pi"]

        # if we want div and sub
        # if isinstance(expr, sp.Mul) and len(expr.args)==2:
        #    if isinstance(expr.args[0], sp.Mul) and isinstance(expr.args[0].args[0], sp.Pow): return ['div']+self.sympy_to_prefix(expr.args[1])+self.sympy_to_prefix(expr.args[0].args[1])
        #    if isinstance(expr.args[1], sp.Mul) and isinstance(expr.args[1].args[0], sp.Pow): return ['div']+self.sympy_to_prefix(expr.args[0])+self.sympy_to_prefix(expr.args[1].args[1])
        # if isinstance(expr, sp.Add) and len(expr.args)==2:
        #    if isinstance(expr.args[0], sp.Mul) and str(expr.args[0].args[0])=='-1': return ['sub']+self.sympy_to_prefix(expr.args[1])+self.sympy_to_prefix(expr.args[0].args[1])
        #    if isinstance(expr.args[1], sp.Mul) and str(expr.args[1].args[0])=='-1': return ['sub']+self.sympy_to_prefix(expr.args[0])+self.sympy_to_prefix(expr.args[1].args[1])

        # if isinstance(expr, sp.Pow) and str(expr.args[1])=='-1':
        #     return ['inv'] + self.sympy_to_prefix(expr.args[0])

        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)

        # Unknown operator
        return self._sympy_to_prefix(str(type(expr)), expr)

    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Mod: "mod",
        sp.Pow: "pow",
        # Misc
        sp.Abs: "abs",
        sp.sign: "sign",
        sp.Heaviside: "step",
        # Exp functions
        sp.exp: "exp",
        sp.log: "log",
        # Trigonometric Functions
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        # Trigonometric Inverses
        sp.asin: "arcsin",
        sp.acos: "arccos",
        sp.atan: "arctan",
    }
