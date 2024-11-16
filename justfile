# Windows
set shell := ["cmd.exe", "/c"]

check:
    ruff check --fix

test:
    pytest