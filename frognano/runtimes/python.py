"""Python discovery and bootstrap for Leaf tools."""

FIND_PYTHON = (
    "python_bin=$(test -x /usr/bin/python3 && echo /usr/bin/python3 "
    "|| command -v python3 || command -v python)"
)
CHECK_PYTHON = (
    f'{FIND_PYTHON}; test -n "$python_bin" && '
    '"$python_bin" -I -S -c "import sys; sys.exit(sys.version_info < (3, 6))"'
)
INSTALL_PYTHON = (
    "if command -v apt-get >/dev/null 2>&1; then "
    "apt-get update -qq && "
    "DEBIAN_FRONTEND=noninteractive apt-get install "
    "-y -qq --no-install-recommends python3; "
    "elif command -v apk >/dev/null 2>&1; then "
    "apk add --no-cache python3; "
    "else echo 'No supported Python package manager (apt-get or apk)' "
    ">&2; exit 1; fi"
)
