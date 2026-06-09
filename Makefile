PYTHON := .venv/bin/python

# Guard: patterns that must never appear in training code.
# grep exits 0 if a match IS found (bad), 1 if clean.
TRAINING_FILES := scripts/step5_cache_features.py \
                  scripts/step6_train_baseline.py \
                  scripts/step7_train_clarity.py \
                  clarity_vision/train_utils.py

GUARD_PATTERN := part_locs\|part_loc\b\|load_part\|keypoint

# ---------------------------------------------------------------------------
# Guard and structural checks (run these before any step)
# ---------------------------------------------------------------------------

.PHONY: guard
guard:
	@echo "[guard] Checking training files for forbidden part-annotation references..."
	@if grep -rn "$(GUARD_PATTERN)" $(TRAINING_FILES) 2>/dev/null; then \
		echo "[guard] FAIL — training code references part-location data (see above)."; \
		exit 1; \
	fi
	@echo "[guard] PASS — no part-annotation references found in training code."

.PHONY: structure-check
structure-check:
	@echo "[structure-check] Verifying required files exist..."
	@missing=0; \
	for f in \
		clarity_vision/__init__.py \
		clarity_vision/models.py \
		clarity_vision/data.py \
		clarity_vision/evaluation.py \
		clarity_vision/train_utils.py \
		scripts/step1_smoke_test.py \
		scripts/step2_verify_data.py \
		scripts/step3_concepts.py \
		scripts/step4_concept_part_map.py \
		scripts/step5_cache_features.py \
		scripts/step6_train_baseline.py \
		scripts/step7_train_clarity.py \
		scripts/step8_gate_eval.py \
		configs/gate.yaml \
		requirements.txt; do \
		if [ ! -f "$$f" ]; then echo "  MISSING: $$f"; missing=1; fi; \
	done; \
	if [ $$missing -eq 1 ]; then echo "[structure-check] FAIL"; exit 1; fi
	@echo "[structure-check] PASS — all required files present."

# ---------------------------------------------------------------------------
# Steps 1–4  (safe to run locally; no GPU required for 1–4)
# ---------------------------------------------------------------------------

.PHONY: step1
step1: guard structure-check
	$(PYTHON) scripts/step1_smoke_test.py

.PHONY: step2
step2: guard structure-check
	$(PYTHON) scripts/step2_verify_data.py

.PHONY: step3
step3: guard structure-check
	$(PYTHON) scripts/step3_concepts.py

.PHONY: step4
step4: guard structure-check outputs/concepts.json
	$(PYTHON) scripts/step4_concept_part_map.py

outputs/concepts.json:
	$(MAKE) step3

# ---------------------------------------------------------------------------
# Steps 5–8  (cluster / GPU; blocked until concept_part_map.json is frozen)
# ---------------------------------------------------------------------------

outputs/concept_part_map.json:
	@echo "ERROR: outputs/concept_part_map.json not found. Run 'make step4' first."
	@exit 1

.PHONY: step5
step5: guard outputs/concept_part_map.json
	$(PYTHON) scripts/step5_cache_features.py

.PHONY: step6
step6: guard outputs/concept_part_map.json
	$(PYTHON) scripts/step6_train_baseline.py

.PHONY: step7
step7: guard outputs/concept_part_map.json
	$(PYTHON) scripts/step7_train_clarity.py

.PHONY: step8
step8: guard
	$(PYTHON) scripts/step8_gate_eval.py

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test
test: guard
	.venv/bin/pytest tests/ -v
