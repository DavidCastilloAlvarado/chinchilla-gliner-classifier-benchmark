COMPOSE ?= docker compose
PROFILE ?= cpu

.PHONY: build run runbuild stop logs test

build:
	$(COMPOSE) --profile $(PROFILE) build

run:
	$(COMPOSE) --profile $(PROFILE) up

runbuild:
	$(COMPOSE) --profile $(PROFILE) up --build

stop:
	$(COMPOSE) --profile cpu --profile cuda down

logs:
	$(COMPOSE) --profile $(PROFILE) logs -f

test:
	uv run tests
