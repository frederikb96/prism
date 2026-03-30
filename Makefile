.PHONY: dev dev-down dev-logs dev-build prod prod-down prod-logs prod-build test lint typecheck check

# Development
dev:
	podman compose --env-file .dev.env -f docker-compose.dev.yaml up -d

dev-down:
	podman compose -f docker-compose.dev.yaml down

dev-logs:
	podman logs prism-dev

dev-build:
	podman compose --env-file .dev.env -f docker-compose.dev.yaml build

# Production
prod:
	podman compose --env-file .prod.env up -d

prod-down:
	podman compose down

prod-logs:
	podman logs prism

prod-build:
	podman compose --env-file .prod.env build

# Quality checks
test:
	uv run pytest tests/unit/ -q

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/prism/

check: lint typecheck test
