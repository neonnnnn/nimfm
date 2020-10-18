# Package
version       = "0.2.0"
author        = "Kyohei Atarashi"
description   = "nimfm: A library for factorization machines in Nim."
license       = "MIT"
srcDir        = "src"


# Dependencies
requires "nim >= 1.0.6", "cligen >= 0.9.43", "nimlapack >= 0.2.0"

# Compile and create binary in ./bin for end users
task make, "builds nimfm":
  exec "mkdir -p bin"
  exec "nim c  -o:bin/nimfm -d:release -d:danger ./src/nimfm.nim"
  exec "nim c  -o:bin/nimfm_cfm -d:release -d:danger ./src/nimfm_cfm.nim"
  exec "nim c  -o:bin/nimfm_sparsefm -d:release -d:danger ./src/nimfm_sparsefm.nim"

