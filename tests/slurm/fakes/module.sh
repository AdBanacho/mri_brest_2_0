# Fake environment-modules `module` function (source, then `export -f module`).
# Records the PYTHONNOUSERSITE seen at load time and then RESETS it, like a
# module that rewrites the Python environment; the controller must re-export it.
module() {
  printf '%s PYTHONNOUSERSITE=%s\n' "$*" "${PYTHONNOUSERSITE:-unset}" >> "${FAKE_MODULE_LOG:?}"
  [[ "${FAKE_MODULE_FAIL:-0}" == 0 ]] || return 1
  export PYTHONNOUSERSITE=0
}
