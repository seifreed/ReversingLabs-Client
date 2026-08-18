"""Everything that decides how a value looks on a terminal or in a document.

Only ``cli`` may read this package. Nothing here reads a ``Settings``, calls
a service or writes a file: a renderer is handed the facts and returns or
prints them, so ``config`` and ``services`` never pull Rich in to answer a
question about a value.
"""
