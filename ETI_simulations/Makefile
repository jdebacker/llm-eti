.PHONY: install run-simulation clean test-simulation run-simulation-high-income run-simulation-4o

install:
	pip install -r ../requirements.txt

run-simulation:
	python cli.py run-simulation

test-simulation:
	python cli.py run-simulation \
		--min-income 50000 \
		--max-income 200000 \
		--income-step 50000 \
		--rate-step 0.05 \
		--responses-per-rate 3 \
		--model gpt-4o-mini

run-simulation-high-income:
	python cli.py run-simulation \
		--min-income 200000 \
		--max-income 1000000 \
		--income-step 50000

run-simulation-4o:
	python cli.py run-simulation --model gpt-4o

.PHONY: run-4o analyze-both

run-4o:
	python cli.py run-simulation \
		--model gpt-4o \
		--output simulation_4o

analyze-both:
	python combine_analyze.py \
		results/simulation_20241105_143622 \
		results/simulation_4o

clean:
	rm -rf results/*