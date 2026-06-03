PYTHON := python
CELERY := celery -A videorag

.PHONY: help reset start start-celery dedupe-videos reindex

help:
	@echo ""
	@echo "  make reset          wipe DB, migrations, indexes (videos in data/videos/ kept)"
	@echo "  make start          run migrations and start Django dev server"
	@echo "  make start-celery   start Celery worker (run in a separate terminal)"
	@echo "  make dedupe-videos  remove byte-identical duplicates in data/videos/ (dry-run)"
	@echo "  make reindex        force re-index all videos after model upgrade"
	@echo ""

# ── Full clean slate ──────────────────────────────────────────────────────────
reset:
	@echo "→ Removing database..."
	rm -f data/db.sqlite3

	@echo "→ Removing migrations (keeping __init__.py)..."
	find search/migrations -name "*.py" ! -name "__init__.py" -delete 2>/dev/null || true

	@echo "→ Removing keyframes and vector index..."
	rm -rf data/keyframes data/index

	@echo "→ Flushing Redis..."
	redis-cli flushall

	@echo ""
	@echo "✓ Clean. Videos in data/videos/ are untouched."
	@echo "  Run 'make start' to begin fresh."

# ── Start Django server ───────────────────────────────────────────────────────
start:
	$(PYTHON) manage.py makemigrations search
	$(PYTHON) manage.py migrate
	$(PYTHON) manage.py runserver

# ── Start Celery worker ───────────────────────────────────────────────────────
start-celery:
	$(CELERY) worker --loglevel=info

dedupe-videos:
	./scripts/dedupe_videos_by_checksum.sh --dry-run

reindex:
	$(PYTHON) manage.py index_videos --force
