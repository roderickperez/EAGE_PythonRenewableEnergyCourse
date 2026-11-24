"""Inspect Python imports and notebook commands without executing learner code."""
import ast
import io
import token
import tokenize


def inspect_source(source):
    """Return static top-level module names and whether a line is a magic command.

    Strings and comments are tokenized as such; a percent operator inside brackets
    is valid Python and is not a command. Dynamic imports are intentionally outside
    static inspection and should be imported explicitly when a package is needed.
    """
    depth = 0
    magic = False
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (token.OP, token.ERRORTOKEN):
                if tok.string in ("%", "!") and depth == 0 and not tok.line[:tok.start[1]].strip():
                    magic = True
                if tok.string in ("(", "[", "{"):
                    depth += 1
                elif tok.string in (")", "]", "}"):
                    depth = max(0, depth - 1)
    except (tokenize.TokenError, IndentationError):
        # Execution reports syntax problems with Python's normal diagnostics.
        pass
    imports = set()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        pass
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])
    return {"magic": magic, "imports": sorted(imports)}
