# Test App

## Current functionality

Benchmarking and Testing

## Options

```commandline
--list               or -l                 print available providers
--benchmark          or -b                 run benchmark
--core               or -c                 test the core library use case
--test               or -t                 run basic test
--test-vector-check                        run a CPU test and compare with test-vector
--test-pow           or -tp                test pow computation
--unit-tests         or -u                 run unit tests
--integration-tests  or -i                 run integration tests
--label-size         or -s <1-256>         set label size [1-256]
--labels-count       or -n <1-32M>         set labels count [up to 32M]
--reference-provider or -r <id>            the result of this provider will be used as a reference [default - CPU]
--print              or -p                 print detailed data comparison report for incorrect results
--pow-diff           or -d <0-256>         count of leading zero bits in target D value [default - 16]
--srand-seed         or -ss <unsigned int> set srand seed value for POW test: 0 - use zero id/seed [default], -1 - use random value
--solution-idx       or -si <unsigned int> set solution index for POW test: index will be compared to be the found solution for Pow [default - unset]
```
