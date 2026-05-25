data:
	python scripts/download_data.py
	python scripts/prepare_sources.py
	python scripts/build_dataset.py

validate:
	python scripts/validate_dataset.py

clean:
	@echo "Placeholder clean command"
