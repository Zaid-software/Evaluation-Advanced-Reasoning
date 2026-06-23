import ast
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorError(Exception):
    pass


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise CalculatorError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise CalculatorError(f"Unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](_eval_node(node.operand))
    raise CalculatorError(f"Unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> float:
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"Could not parse expression '{expression}': {e}")
    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        raise CalculatorError(f"Division by zero in expression '{expression}'")
    return float(result)


if __name__ == "__main__":
    # quick self-test
    tests = [
        ("3 * 24 - 18", 54.0),
        ("(15 + 8) * 2 * 7", 322.0),
        ("500 * 0.6", 300.0),
        ("300 / 23", 300 / 23),
    ]
    for expr, expected in tests:
        got = calculator(expr)
        status = "OK" if abs(got - expected) < 1e-6 else "FAIL"
        print(f"[{status}] calculator('{expr}') = {got} (expected {expected})")
