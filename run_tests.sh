#!/bin/bash

# Set output directory for logs and results
LOGDIR="logs/results"
mkdir -p "$LOGDIR"

# Clean up old logs in the output directory
rm -f "$LOGDIR"/*.log

# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Running inference tests..."
pytest tests/inference/ -v > "$LOGDIR/inference_tests.log" 2>&1

echo "Running judge tests..."
pytest tests/judge/ -v > "$LOGDIR/judge_tests.log" 2>&1

echo "Running analytics component tests..."
pytest tests/analytics/ -v > "$LOGDIR/analytics_component_tests.log" 2>&1
pytest tests/test_analytics.py tests/test_analytics_smoke.py -v > "$LOGDIR/analytics_api_tests.log" 2>&1

echo "Running visualization tests..."
pytest tests/analytics/test_visualization.py -v > "$LOGDIR/visualization_tests.log" 2>&1
pytest tests/test_visualization_smoke.py -v > "$LOGDIR/visualization_smoke_test.log" 2>&1
pytest tests/analytics/test_visualization_cli.py -v > "$LOGDIR/visualization_cli_tests.log" 2>&1

echo "Running end-to-end tests..."
pytest tests/test_end_to_end.py -v > "$LOGDIR/end_to_end_tests.log" 2>&1

echo "Generating coverage reports..."
pytest --cov=src.inference tests/inference/ > "$LOGDIR/inference_coverage.log" 2>&1
pytest --cov=src.judge tests/judge/ > "$LOGDIR/judge_coverage.log" 2>&1
pytest --cov=src.analytic tests/analytics/ tests/test_analytics.py > "$LOGDIR/analytics_coverage.log" 2>&1
pytest --cov=src.analytics.visualization tests/analytics/test_visualization*.py tests/test_visualization_smoke.py > "$LOGDIR/visualization_coverage.log" 2>&1

echo "All tests complete! Logs and results are in $LOGDIR"
