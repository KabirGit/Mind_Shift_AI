.PHONY: install install-dev run seed test lint typecheck clean help

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:        ## Install runtime dependencies
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

install-dev:    ## Install all dependencies including dev tools
	pip install -r requirements-dev.txt
	python -m spacy download en_core_web_sm

run:            ## Start the Streamlit app
	streamlit run app.py

seed:           ## Populate demo data (deletes existing DB and FAISS store first)
	@echo "Seeding demo data..."
	python scripts/seed_demo_data.py

test:           ## Run all tests
	pytest -v

lint:           ## Run ruff linter
	ruff check .

typecheck:      ## Run mypy type checker
	mypy backend/ --ignore-missing-imports --no-strict-optional

clean:          ## Remove generated data files (keeps .env and source code)
	rm -f data/journal.db data/latency_log.jsonl
	rm -rf faiss_store/
	@echo "Cleaned. Run 'make seed' to repopulate demo data."
