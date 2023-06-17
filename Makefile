default: | help

SHELL=/bin/bash
COMPE=google-research-identify-contrails-reduce-global-warming

setup: ## setup install packages
	# bootstrap of rye
	if ! command -v rye > /dev/null 2>&1; then \
		curl -sSf https://rye-up.com/get | bash; \
		echo 'source $HOME/.rye/env' > ~/.profile; \
	fi;
	rye sync

download_data: ## download data from competition page
	if [[ ! -d ./input ]]; then \
		mkdir ./input; \
	fi;
	rye add --dev kaggle \
	&& rye run kaggle competitions download -c "${COMPE}" -p ./input; \
	unzip "./input/${COMPE}.zip" -d "./input/${COMPE}"

lint: ## lint code
	rye run ruff check scripts src

mypy: ## typing check
	rye run mypy --config-file pyproject.toml scirpts src

format: ## auto format
	rye run isort scripts src
	rye run black scripts src

test: ## run test with pytest
	rye run pytest -c tests

clean:
	rm -rf ./output/* wandb/*

help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
