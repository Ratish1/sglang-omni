FILES="tests/unit_test/pipeline/test_session_graph.py tests/unit_test/benchmarks/test_duplex_legacy.py tests/unit_test/benchmarks/test_duplex_profiles.py tests/unit_test/benchmarks/test_duplex_turn_based.py tests/unit_test/benchmarks/test_duplex_v15_runner.py"
for repetition in 1 2 3 4 5; do
  for tree in main08 fix04; do
    cd /data/ab/$tree && PYTHONPATH=/data/ab/$tree python3 -m pytest -q -p no:cacheprovider $FILES > /data/checks/flaky_${tree}_$repetition.log 2>&1
    echo "$tree $repetition: $(grep -E "passed|failed" /data/checks/flaky_${tree}_$repetition.log | tail -1) $(grep -E "^FAILED" /data/checks/flaky_${tree}_$repetition.log | sed 's/.*:://; s/ - .*//' | tr '\n' ' ')"
  done
done
echo FLAKY_DONE
