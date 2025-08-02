.PHONY: install run-simulation clean test-simulation run-simulation-high-income run-simulation-4o run-simple-regression book format lint

# Install dependencies using uv
install:
	@command -v uv >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv sync

# Legacy simulation commands
run-simulation:
	uv run python cli.py run-simulation

test-simulation:
	uv run python cli.py run-simulation \
		--min-income 50000 \
		--max-income 200000 \
		--income-step 50000 \
		--rate-step 0.05 \
		--responses-per-rate 3 \
		--model gpt-4o-mini

run-simulation-high-income:
	uv run python cli.py run-simulation \
		--min-income 200000 \
		--max-income 1000000 \
		--income-step 50000

run-simulation-4o:
	uv run python cli.py run-simulation --model gpt-4o

.PHONY: run-4o analyze-both

run-4o:
	uv run python cli.py run-simulation \
		--model gpt-4o \
		--output simulation_4o

analyze-both:
	uv run python combine_analyze.py \
		results/simulation_20241105_143622 \
		results/simulation_4o

run-simple-regression:
	uv run python simple_regression.py

# JupyterBook commands
book:
	cd book && make book

book-serve:
	cd book && make serve

book-pdf:
	cd book && make pdf

book-test:
	cd book && make test-data && make figures && make tables && make book

clean:
	rm -rf results/*
	cd book && make clean-all

# Format code
format:
	uv run black .
	uv run ruff check --fix .

# Lint code
lint:
	uv run black --check .
	uv run ruff check .
	uv run mypy . --ignore-missing-imports

# Help
help:
	@echo "Simulation commands:"
	@echo "  install               - Install dependencies with uv"
	@echo "  run-simulation        - Run standard simulation"
	@echo "  test-simulation       - Run quick test simulation"
	@echo "  run-simple-regression - Run enhanced regression analysis"
	@echo ""
	@echo "JupyterBook commands:"
	@echo "  book                  - Build the JupyterBook"
	@echo "  book-serve            - Serve book locally"
	@echo "  book-pdf              - Generate PDF"
	@echo "  book-test             - Run test pipeline with book"
	@echo ""
	@echo "Development commands:"
	@echo "  format                - Format code with black and ruff"
	@echo "  lint                  - Run linters"
	@echo ""
	@echo "  clean                 - Clean all generated files"