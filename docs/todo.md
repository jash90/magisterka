# Bug Fixes & Feature Additions

## 🔴 Bugs / Things to Fix

### Backend

#### ~~1. SHAP global cache fails to initialize~~ ✅ DONE
- **Fix applied:** `get_tree_model()` helper selects random_forest/xgboost as TreeExplainer model; KernelExplainer fallback with 50 samples on failure.

#### ~~2. DALEX variable importance not cached — returns fake data~~ ✅ DONE
- **Fix applied:** `y_train` converted to `np.float64` before passing to DALEXWrapper.

#### ~~3. DALEX Break Down per instance — silent fallback with no user feedback~~ ✅ DONE
- **Fix applied:** `DALEXExplanation` schema has `fallback_reason: Optional[str]`; populated on exception.

#### ~~4. `/explain/patient` always uses demo explanation~~ ✅ DONE
- **Fix applied:** Calls `_get_real_shap_explanation()` before demo fallback.

#### ~~5. XAI for dialysis — feature name mismatch~~ ✅ DONE
- **Fix applied:** `task_type` parameter added to all XAI endpoints and `ExplanationRequest` schema. `get_xai_context(task_type)` routes to correct model/features. Dashboard passes `task_type: "dialysis"` and `dialysis_patient` when in dialysis mode.

### Frontend (Streamlit)

#### ~~6. Welcome page lists "LightGBM" — model does not exist~~ ✅ DONE
- **Fix applied:** Replaced with correct list: XGBoost, Random Forest, Calibrated SVM, Naive Bayes, Stacking Ensemble.

#### ~~7. Missing validation for `opoznienie_rozpoznia`~~ ✅ DONE
- **Fix applied:** `@field_validator` in `PatientInput` and `max_value=wiek` in Streamlit form.

#### ~~8. `process_batch_patients` reads `prediction_task` from global scope~~ ✅ DONE
- **Fix applied:** `prediction_task` and `selected_model_type` passed as parameters.

#### ~~9. Hardcoded `"http://localhost:8000"` as `API_URL`~~ ✅ DONE
- **Fix applied:** `API_URL = os.getenv("API_URL", "http://localhost:8000")` with `import os`.

#### ~~10. XAI tabs show stale results after form data changes~~ ✅ DONE
- **Fix applied:** `analyzed` session state reset when patient form hash changes.

---

## 🟡 Features to Add / Extend

### Backend

#### ~~11. Dedicated XAI endpoints for dialysis~~ ✅ DONE
- **Fix applied:** `task_type` parameter added to all 9 XAI endpoints. `_initialize_dialysis_xai_caches()` initializes dialysis SHAP and DALEX caches in background thread.

#### 12. `/explain/ebm/local` for dialysis
- **Status:** Partial. EBM endpoint accepts `task_type`. Training script has EBM section but model not trained yet (requires running `scripts/train_dialysis_model.py` with EBM enabled).

#### ~~13. LLM integration for the chat~~ ✅ DONE
- **Fix applied:** `_generate_openai_response()` when `OPENAI_API_KEY` set (GPT-4o-mini); falls back to rule-based.

#### ~~14. Diagnostic endpoint `/health/models`~~ ✅ DONE
- **Fix applied:** `/health/models` endpoint showing model load status, metrics, XAI cache info.

#### ~~15. Feature name consistency validation at startup~~ ✅ DONE
- **Fix applied:** `startup_event()` validates `feature_names.json` vs `model.n_features_in_`; logs warning on mismatch.

#### ~~16. Timeout for per-instance XAI calls~~ ✅ DONE
- **Fix applied:** `asyncio.wait_for` 30s timeout via `_run_in_executor_with_timeout`; falls back to demo on timeout.

#### ~~17. Model calibration metrics~~ ✅ DONE
- **Fix applied:** `/models/calibration` endpoint computes Brier score and `calibration_curve` per model using sklearn; returns `CalibrationResponse` schema.

### Frontend (Streamlit)

#### ~~18. XAI tabs for dialysis mode~~ ✅ DONE
- **Fix applied:** All XAI API calls pass `task_type: _xai_task_type` and `dialysis_patient: patient_data` when in dialysis mode.

#### ~~19. Calibration curve (reliability diagram)~~ ✅ DONE
- **Fix applied:** "Kalibracja" tab added with Brier score metrics and reliability diagram chart (plotly).

#### ~~20. PDF report export~~ ✅ DONE
- **Fix applied:** "Generuj raport PDF" button added after tabs; uses `fpdf2` to generate patient data, prediction, risk factors, and medical disclaimer.

#### ~~21. Multi-patient comparison mode~~ ✅ DONE
- **Fix applied:** Expandable "Porównanie pacjentów" section with forms for patients 2 and 3; side-by-side metric display.

#### ~~22. Persist model selection across sessions~~ ✅ DONE
- **Fix applied:** `key="selected_model"` added to model selectbox; Streamlit auto-persists in session_state.

#### ~~23. Visible demo mode warning in XAI tabs~~ ✅ DONE
- **Fix applied:** `st.warning("⚠️ Wyniki poniżej są symulowane...")` shown when XAI returns no data.

#### ~~24. Error handling for network failures in batch mode~~ ✅ DONE
- **Fix applied:** `call_batch_api` stores `error_count`/`success_count`/`errors` in session_state; batch results section shows `st.warning` with counts and expandable error details.

---

## 🟢 Minor Quality Fixes

| # | Location | Status | Description |
|---|---|---|---|
| ~~25~~ | `schemas.py` | ✅ DONE | `FeatureContribution.direction` → `Literal["increases_risk", "decreases_risk"]` |
| ~~26~~ | `main.py` | ✅ DONE | `@app.on_event("startup")` → `lifespan` context manager |
| ~~27~~ | `streamlit_app.py` | ✅ DONE | Added `import os` for `os.getenv` |
| ~~28~~ | `streamlit_app.py` | ✅ DONE | `width='stretch'` → `use_container_width=True` |
| 29 | `main.py` | RESOLVED (not a bug) | `get_demo_dialysis_prediction` — cross-file duplication between main.py and streamlit_app.py with different signatures; expected behavior |
| ~~30~~ | `schemas.py` | ✅ DONE | `BatchPatientInput` Pydantic v2: `min_items/max_items` → `min_length/max_length` |

---

## 🟢 New Bugs Fixed (Session 3 — Dialysis task_type migration gaps)

| # | Location | Status | Description |
|---|---|---|---|
| ~~31~~ | `main.py`, `streamlit_app.py` | ✅ DONE | BUG A: `/explain/comparison/global` missing `task_type` param — added param, use `get_xai_context(task_type)`, dashboard passes `?task_type={_xai_task_type}` |
| ~~32~~ | `main.py` | ✅ DONE | BUG B: `/predict/all-models` has no dialysis support — added `/predict/dialysis/all-models` endpoint; dashboard branches on `_xai_task_type` |
| ~~33~~ | `main.py` | ✅ DONE | BUG C: DALEX `y_train` not converted to `float64` in comparison and PDP endpoints — wrapped with `np.array(..., dtype=np.float64)` |
| ~~34~~ | `streamlit_app.py` | ✅ DONE | BUG D: Multi-patient comparison always calls `/predict` (mortality) — now branches on `prediction_task` to call `/predict/dialysis` for dialysis mode |
| ~~35~~ | `schemas.py`, `main.py`, `streamlit_app.py` | ✅ DONE | BUG E: Chat endpoint always uses mortality prediction — added `task_type` to `ChatRequest`, chat handler branches on `task_type`, dashboard passes `task_type` in `chat_data` |
| ~~36~~ | `streamlit_app.py` | ✅ DONE | BUG F: PDF export uses only demo explanation factors — SHAP result stored in `st.session_state["shap_api_result"]` after SHAP tab call; PDF export uses real SHAP data if available |
| ~~37~~ | `schemas.py`, `main.py` | ✅ DONE | BUG G: `/explain/patient` missing `task_type` support — added `task_type` and `dialysis_patient` to `PatientExplanationRequest`; endpoint branches on `task_type` for prediction and SHAP |
| ~~38~~ | `schemas.py` | ✅ DONE | BUG H: `.dict()` deprecated in Pydantic v2 — replaced with `.model_dump()` in `dialysis_patient_to_array()` and `patient_to_array()` |
| ~~12~~ | `scripts/train_dialysis_model.py` | ✅ DONE | Item #12 (partial fix): EBM save path changed from `ebm_model.joblib` → `dialysis_ebm_model.joblib` to match API expectation (`main.py:286`) |

---

## 🟢 Codebase Audit Fixes (Session 4 — Bugs #39-63)

### Phase 1: Backend Critical Fixes

| # | Location | Status | Description |
|---|---|---|---|
| ~~39~~ | `main.py` | ✅ DONE | `.dict()` → `.model_dump()` in both `http_exception_handler` and `general_exception_handler` |
| ~~40~~ | `schemas.py`, `main.py` | ✅ DONE | Chat dialysis: added `dialysis_patient: Optional[DialysisPatientInput]` to `ChatRequest`; chat endpoint uses `request.dialysis_patient` when `task_type=="dialysis"` |
| ~~41~~ | `main.py` | ✅ DONE | `_get_contextual_factors()` now accepts `task_type` and `dialysis_patient` params; uses `get_xai_context(task_type)` internally |
| ~~42~~ | `schemas.py` | ✅ DONE | `RiskFactorItem.direction` typed as `Literal["increases_risk", "decreases_risk", "neutral"]` |
| ~~43~~ | `main.py` | ✅ DONE | Removed dead `_comparison_global_cache` declaration |

### Phase 2: XAI Module Fixes

| # | Location | Status | Description |
|---|---|---|---|
| ~~44~~ | `ebm_explainer.py` | ✅ DONE | Polish direction strings → English: `'increases_risk'` / `'decreases_risk'` |
| ~~45~~ | `main.py` | ✅ DONE | Dialysis EBM init now passes `class_names=['Brak dializy', 'Dializa']` |
| ~~46~~ | `dalex_wrapper.py` | ✅ DONE | `self.y = np.asarray(y, dtype=np.float64) if y is not None else y` (defensive float64 conversion) |

### Phase 3: Dashboard Critical Fixes

| # | Location | Status | Description |
|---|---|---|---|
| ~~47~~ | `streamlit_app.py` | ✅ DONE | `get_demo_dialysis_prediction()` now returns `"prediction": int(probability > 0.5)` key |
| ~~48~~ | `streamlit_app.py` | ✅ DONE | PDF export `task_label`: fixed comparison from ASCII `"Smiertelnosc"` to `prediction_task == "Śmiertelność"` |
| ~~49~~ | `streamlit_app.py` | ✅ DONE | Batch dialysis fallback now tries `/predict/dialysis` API before demo |
| ~~50~~ | `streamlit_app.py` | ✅ DONE | Added `get_demo_dialysis_explanation()` (kreatynina-weighted); used at batch loop and single-patient analysis |

### Phase 4: Dashboard Label Fixes

| # | Location | Status | Description |
|---|---|---|---|
| ~~51~~ | `streamlit_app.py` | ✅ DONE | `create_probability_histogram` and `create_age_risk_scatter` accept `prediction_task` param; call sites updated |
| ~~52~~ | `streamlit_app.py` | ✅ DONE | Multi-model bar chart title/axis uses `outcome_label_mm` conditional on `_xai_task_type` |
| ~~53~~ | `streamlit_app.py` | ✅ DONE | Multi-model Sensitivity description: "przypadków zgonu" vs "przypadków potrzeby dializy" |
| ~~54~~ | `streamlit_app.py` | ✅ DONE | Chat fallback: `risk_level` translated via `risk_map` before display |
| ~~55~~ | `streamlit_app.py` | ✅ DONE | `applymap` → `map` (pandas deprecation) |
| ~~56~~ | `streamlit_app.py` | ✅ DONE | Copyright updated: `© 2024` → `© 2024-2026` |

### Phase 5: Config, Scripts, Tests

| # | Location | Status | Description |
|---|---|---|---|
| ~~57~~ | `tests/test_models.py` | ✅ DONE | `test_model_config_structure`: assertion allows `'class' in config or 'factory' in config` |
| ~~58~~ | `src/models/config.py` | ✅ DONE | Removed deprecated `'use_label_encoder': False` from xgboost `base_params` and stacking ensemble |
| ~~59~~ | `scripts/train_dialysis_model.py` | ✅ DONE | `import numpy as np` moved to top-level; removed redundant import in `__main__` block |
| ~~60~~ | `scripts/train_model.py` | ✅ DONE | Hardcoded absolute path replaced with `f"{project_root}"` |

### Phase 6: Low-Priority Backend Enhancements

| # | Location | Status | Description |
|---|---|---|---|
| ~~61~~ | `main.py` | ✅ DONE | `/model/global-importance` accepts `task_type` param; uses `get_xai_context(task_type)` caches |
| ~~62~~ | `main.py` | ✅ DONE | `/model/info` accepts `task_type` param; branches for dialysis model info |
| ~~63~~ | `main.py` | ✅ DONE | `/health/xai` and `/health/models` report `dialysis_shap_cached`, `dialysis_dalex_cached`, `dialysis_ebm_cached` |

---

## Session 5 — Comprehensive Audit #64-87

### Phase 1: CRITICAL — LIME Integration

| # | Location | Status | Description |
|---|---|---|---|
| ~~64~~ | `main.py` | ✅ DONE | `_get_real_lime_explanation`: `feature_weights` is list of tuples; fixed indexing from `.get()` to `fw[0]`/`fw[1]` |
| ~~65~~ | `main.py` | ✅ DONE | LIME result uses `"probability_positive"` key (not `"probability"`); fixed with fallback chain |
| ~~66~~ | `main.py` | ✅ DONE | Demo confidence interval clamped: `max(0.0, prob-0.1)` / `min(1.0, prob+0.1)` |

### Phase 2: CRITICAL — Dashboard Data Flow

| # | Location | Status | Description |
|---|---|---|---|
| ~~67~~ | `streamlit_app.py` | ✅ DONE | Chat now sends `conversation_history` from `st.session_state.get("chat_history", [])` |
| ~~68~~ | `streamlit_app.py` | ✅ DONE | Chat in dialysis mode now sends `dialysis_patient: patient_data` |
| ~~69~~ | `streamlit_app.py` | ✅ DONE | `prepare_patients_for_batch` accepts `prediction_task`; builds `DialysisPatientInput`-format dict when task is dialysis |
| ~~70~~ | `streamlit_app.py` | ✅ DONE | Fixed Cyrillic `а` (U+0430) → Latin `a` in PDF direction label |

### Phase 3: MEDIUM — Backend Functional Gaps

| # | Location | Status | Description |
|---|---|---|---|
| ~~71~~ | `main.py` | ✅ DONE | `predict_dialysis_all_models`: real models now include metrics from `dialysis_model_metadata` |
| ~~72~~ | `main.py` | ✅ DONE | `generate_contextual_response` accepts `task_type`; branches for dialysis-specific recommendations and default response |
| ~~73~~ | `streamlit_app.py` | ✅ DONE | LIME tab color description now uses `_lime_outcome` conditional on `_xai_task_type` |
| ~~74~~ | `main.py` | ✅ DONE | `get_global_importance(task_type)` returns dialysis-specific features when no cache is available |
| ~~75~~ | `main.py` | ✅ DONE | `XAI_TIMEOUT_SECONDS` reads from `os.getenv("XAI_TIMEOUT", "30")` |
| ~~76~~ | `schemas.py` | ✅ DONE | `DialysisPatientInput` docstring documents which fields are model inputs vs. metadata |

### Phase 4: MEDIUM — Dashboard Quality

| # | Location | Status | Description |
|---|---|---|---|
| ~~77~~ | `streamlit_app.py` | ✅ N/A | `get_demo_explanation` already only called when `prediction_task == "Śmiertelność"` (no change needed) |
| ~~78~~ | `streamlit_app.py` | ✅ DONE | Welcome page model list calls `get_available_models()` at render time; falls back to static list if API unavailable |
| ~~79~~ | `streamlit_app.py` | ✅ DONE | PDF export adds `needs_dialysis` field when `prediction_task == "Potrzeba dializy"` |
| ~~80~~ | `streamlit_app.py` | ✅ DONE | Batch processing mode description now conditional: "modelu XGBoost" vs "modeli dializy" |

### Phase 5: LOW — Robustness & Deprecations

| # | Location | Status | Description |
|---|---|---|---|
| ~~81~~ | `main.py` | ✅ N/A | All XAI except blocks already have `logger.warning/error` — no silent handlers found |
| ~~82~~ | `comparison.py` | ✅ N/A | No Polish direction strings found — no change needed |
| ~~83~~ | `lime_explainer.py` | ✅ DONE | `to_json()` direction strings: `'zwiększa ryzyko'` → `'increases_risk'`, etc. |
| ~~84~~ | `requirements.txt` | ✅ DONE | Added `scipy>=1.10.0` (used in comparison.py but was not listed) |
| ~~85~~ | `preprocessing.py` | ✅ N/A | No `dtype == 'object'` comparison found; code uses `select_dtypes` — no change needed |
| ~~86~~ | `streamlit_app.py` | ✅ DONE | `COLUMN_MAPPING` extended with dialysis-specific aliases (oddechowy, pulsy, hospital, etc.) |
| ~~87~~ | `config.py` | ✅ N/A | No `ModelType` enum exists in `config.py` — no change needed |

---

## Session 6 — Comprehensive Audit #88-121

### Phase 1: CRITICAL — Event Loop Blocking (main.py)

| # | Location | Status | Description |
|---|---|---|---|
| ~~88~~ | `main.py:/explain/comparison` | ✅ DONE | Wrapped `_get_real_shap_explanation`, `_get_real_lime_explanation`, and DALEX Break Down in `_run_in_executor_with_timeout` |
| ~~89~~ | `main.py:/explain/dalex` | ✅ DONE | Wrapped `wrapper.explain_instance_break_down(instance)` in `_run_in_executor_with_timeout` |
| ~~90~~ | `main.py:/explain/patient` | ✅ DONE | Wrapped `_get_real_shap_explanation(...)` in `_run_in_executor_with_timeout` |
| ~~91~~ | `main.py:_generate_openai_response` | ✅ DONE | Wrapped `client.chat.completions.create()` in `asyncio.get_running_loop().run_in_executor(None, ...)` |
| ~~92~~ | `main.py:chat` | ✅ DONE | Wrapped `_get_contextual_factors(...)` call in `_run_in_executor_with_timeout` |

### Phase 2: CRITICAL — Dashboard Data Flow (streamlit_app.py)

| # | Location | Status | Description |
|---|---|---|---|
| ~~93~~ | `streamlit_app.py:~2775` | ✅ DONE | Changed `st.session_state.get("chat_history", [])` → `st.session_state.get("messages", [])` |
| ~~94~~ | `streamlit_app.py:~1810` | ✅ DONE | Clamped: `opoznienie_rozpoznia = max(0, wiek - wiek_rozpoznania)` |
| ~~95~~ | `streamlit_app.py:~1671` | ✅ DONE | Removed unused `demo_enabled` checkbox — was never wired into logic |
| ~~96~~ | `schemas.py:PatientInput` | ✅ DONE | Added `extra = "ignore"` to `class Config` to silently drop unknown dialysis fields |

### Phase 3: MEDIUM — Backend Fallback Bugs (main.py)

| # | Location | Status | Description |
|---|---|---|---|
| ~~97~~ | `main.py` (8+ sites) | ✅ DONE | Added `task_type=task_type` to all `get_global_importance()` calls in SHAP global, DALEX VI, DALEX PDP, EBM global, EBM feature function, comparison global |
| ~~98~~ | `main.py:get_global_importance` | ✅ DONE | Updated dialysis hardcoded fallback dict: removed `Max_CRP`, `Manifestacja_Sercowo-Naczyniowy`, `Powiklania_Serce/pluca`; added `Zaostrz_Wymagajace_Hospital`, `Manifestacja_Neurologiczny`, `Powiklania_Infekcja` |
| ~~99~~ | `main.py` (4 sites) | ✅ DONE | Added `_get_demo_dialysis_explanation()` helper; all 4 demo fallback sites check `task_type` |
| ~~100~~ | `main.py:~840` | ✅ DONE | `asyncio.get_event_loop()` → `asyncio.get_running_loop()` |
| ~~101~~ | `main.py:~841` | ✅ DONE | Created module-level `_XAI_EXECUTOR = ThreadPoolExecutor(max_workers=2)`; removed per-call executor |
| ~~102~~ | `main.py:FEATURE_TRANSLATIONS` | ✅ DONE | Added `Manifestacja_Oddechowy`, `Zaostrz_Wymagajace_Hospital`, `Pulsy` to translations and field_map |
| ~~103~~ | `main.py:feature_keywords` | ✅ DONE | Added `"pulsy"`, `"oddechow"`, `"hospital"` entries to chat feature keywords |
| ~~104~~ | `main.py:~2476` | ✅ DONE | Removed dead underscore variant `"Manifestacja_Sercowo_Naczyniowy"` from translations dict |

### Phase 4: MEDIUM — Dashboard Functional Gaps (streamlit_app.py)

| # | Location | Status | Description |
|---|---|---|---|
| ~~105~~ | `streamlit_app.py:~2520` | ✅ DONE | `/model/info` now called with `?task_type={_xai_task_type}` for PDP feature list |
| ~~106~~ | `streamlit_app.py:~800` | ✅ DONE | Mortality batch fallback now tries `/predict` API before falling back to demo |
| ~~107~~ | `streamlit_app.py` (4 sites) | ✅ DONE | All 4 bare `except:` → `except Exception:` |
| ~~108~~ | `streamlit_app.py` | ✅ DONE | `batch_results`, `comp_pred_2`, `comp_pred_3`, `analyzed` cleared when `prediction_task` changes |
| ~~109~~ | `streamlit_app.py:~1929` | ✅ DONE | Replaced hardcoded `"XGBoost"` with `MODEL_DISPLAY_NAMES.get(selected_model_type, ...)` |
| ~~110~~ | `main.py:~2113` | ✅ DONE | Removed unused variable `n_features = len(importance)` |

### Phase 5: MEDIUM — XAI / Schema / Config Fixes

| # | Location | Status | Description |
|---|---|---|---|
| ~~111~~ | `dalex_wrapper.py:~76` | ✅ DONE | Changed `y=y` → `y=self.y` so float64-converted array is passed to Explainer |
| ~~112~~ | `lime_explainer.py:~325` | ✅ DONE | Filter `None` values before `max(set(...))` in `_calculate_ranking_consistency` |
| ~~113~~ | `schemas.py:patients_to_matrix` | ✅ DONE | Changed `dtype=np.float32` → `dtype=np.float64` |
| ~~114~~ | `schemas.py:PatientInput` | ✅ DONE | Added `extra = "ignore"` to `class Config` (Pydantic v2 approach) |
| ~~115~~ | `config.py:logistic_regression` | ✅ DONE | Removed `'elasticnet'` from `random_search` penalties (requires incompatible `l1_ratio`/`solver`) |

### Phase 6: LOW — Cleanup & Robustness

| # | Location | Status | Description |
|---|---|---|---|
| ~~116~~ | `shap_explainer.py:plot_waterfall,plot_beeswarm` | ✅ DONE | Removed unused `fig, ax = plt.subplots()` before SHAP creates its own figure |
| ~~117~~ | `streamlit_app.py:~846` | ✅ DONE | Removed dead `parse_large_file_streaming` function (never called) |
| ~~118~~ | `streamlit_app.py:DEFAULT_VALUES` | ✅ DONE | Added `manifestacja_oddechowy`, `zaostrz_wymagajace_hospital`, `pulsy` to defaults |
| ~~119~~ | `main.py:run_server` | ✅ DONE | `reload=True` → `reload=os.getenv("DEBUG", "false").lower() == "true"` |
| ~~120~~ | `shap_explainer.py:explain_dataset` | ✅ DONE | Removed redundant `if/else` branches that called identical `shap_values(X_sample)` |
| ~~121~~ | `docs/todo.md` | ✅ DONE | Updated with Session 6 audit summary |

---

## Session 7 — Comprehensive Audit #122-152

### Phase 1: CRITICAL — Security Fixes (main.py)

| # | Location | Status | Description |
|---|---|---|---|
| ~~122~~ | `main.py:CORS` | ✅ DONE | `allow_origins=["*"]` → `os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")` |
| ~~123~~ | `main.py:~20 sites` | ✅ DONE | All `raise HTTPException(status_code=500, detail=str(e))` → `detail="Wewnętrzny błąd serwera"`; `general_exception_handler` no longer leaks `str(exc)` |
| ~~124~~ | `main.py:2965-2967` | ✅ DONE | Prompt injection via `conversation_history` — role validated: only `"user"` or `"assistant"` allowed |

### Phase 2: CRITICAL — Data Flow & Test Coverage

| # | Location | Status | Description |
|---|---|---|---|
| ~~125~~ | `schemas.py + main.py` | ✅ DONE | `DialysisPredictionOutput` now has `prediction: int` field; computed as `int(probability > 0.5)` in predict_dialysis endpoints |
| ~~126~~ | `tests/test_models.py` | ✅ DONE | Added `TestDialysisSchemas` class with 9 tests covering schema validation, array conversion, edge cases |

### Phase 3: MEDIUM — Dialysis Consistency

| # | Location | Status | Description |
|---|---|---|---|
| ~~127~~ | `main.py:SHAP/LIME demo fallback` | ✅ DONE | When `task_type=="dialysis"`, demo fallback now uses `_get_demo_dialysis_explanation()` instead of mortality demo |
| ~~128~~ | `main.py:_get_contextual_factors` | ✅ DONE | Chat contextual factors demo fallback checks `task_type=="dialysis"` before calling `get_demo_explanation` |
| ~~129~~ | `main.py:_get_feature_value_from_patient` | ✅ DONE | Added `dialysis_patient=None` param; checks dialysis_patient first for matching fields |
| ~~130~~ | `main.py:generate_contextual_response` | ✅ DONE | Added `dialysis_patient=None` param; dialysis recommendations read from correct patient object |
| ~~131~~ | `streamlit_app.py:normalize_dataframe` | ✅ DONE | `opoznienie_rozpoznia` computation uses `.clip(lower=0)` to prevent negative values |
| ~~132~~ | `schemas.py:DialysisPatientInput` | ✅ DONE | Added `extra = "ignore"` to `DialysisPatientInput.Config` for consistency with PatientInput |

### Phase 4: MEDIUM — Backend Robustness

| # | Location | Status | Description |
|---|---|---|---|
| ~~133~~ | `config.py:logistic_regression` | ✅ DONE | Removed `'newton-cg'` and `'lbfgs'` from `random_search['solver']` (incompatible with l1 penalty) |
| ~~134~~ | `schemas.py:ChatRequest` | ✅ DONE | `conversation_history` field now has `max_length=20` to prevent memory exhaustion |
| ~~135~~ | `main.py:_generate_openai_response` | ✅ DONE | Created `_OPENAI_EXECUTOR = ThreadPoolExecutor(max_workers=3)`; used instead of default executor |
| ~~136~~ | `main.py:/explain/ebm/local` | ✅ DONE | EBM local dialysis demo now calls `predict_dialysis()` for actual probability instead of hardcoded 0.35 |
| ~~137~~ | `main.py:/predict` | ✅ DONE | Model existence check moved BEFORE `get_model()` call (was unreachable before) |
| ~~138~~ | `main.py:/predict/dialysis/all-models` | ✅ DONE | Guard added: raises HTTP 503 when `dialysis_feature_names` is None |

### Phase 5: MEDIUM — Dashboard Improvements

| # | Location | Status | Description |
|---|---|---|---|
| ~~139~~ | `streamlit_app.py:~2871` | ✅ DONE | Removed unused `comparison_patients` list (dead code) |
| ~~140~~ | `streamlit_app.py:~1509` | ✅ DONE | `"messages"` added to keys cleared on task switch (chat history clears on mortality↔dialysis change) |
| ~~141~~ | `streamlit_app.py:call_batch_api` | ✅ DONE | HTTP errors stored in `st.session_state['_batch_error']` instead of silently swallowed |
| ~~142~~ | `main.py:/predict/all-models` | ✅ DONE | `patient_to_array()` and `X = np.array()` moved outside model loop (was recomputed each iteration) |
| ~~143~~ | `schemas.py:PatientInput` | ✅ DONE | Added `@model_validator(mode='after')` to clamp `wiek_rozpoznania` to `wiek` if it exceeds it |

### Phase 6: LOW — Cleanup & Polish

| # | Location | Status | Description |
|---|---|---|---|
| ~~144~~ | `ebm_explainer.py`, `lime_explainer.py`, `comparison.py` | ✅ DONE | Removed unused `Tuple`, `Union`, `Callable` imports |
| ~~145~~ | `main.py:_generate_openai_response` | ✅ DONE | `model="gpt-4o-mini"` → `model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")` |
| ~~146~~ | `main.py:lifespan` | ✅ DONE | `_XAI_EXECUTOR.shutdown(wait=False)` and `_OPENAI_EXECUTOR.shutdown(wait=False)` added to lifespan shutdown |
| ~~147~~ | `main.py:/explain/ebm/global demo` | ✅ DONE | EBM demo interactions: dialysis uses `"Kreatynina × Manifestacja_Nerki"` instead of `"Manifestacja_Nerki × Dializa"` |
| ~~148~~ | `streamlit_app.py:welcome page` | ✅ DONE | Welcome page reuses `available_models_data` variable instead of making a redundant API call |
| ~~149~~ | `shap_explainer.py:plot_waterfall,plot_beeswarm` | ✅ DONE | Added `plt.close('all')` after returning the figure to prevent matplotlib figure leaks |
| ~~150~~ | `schemas.py:RiskFactorItem` | ✅ DONE | Removed unused `"neutral"` from `direction` Literal (no code ever produces it) |
| ~~151~~ | `comparison.py:_calculate_spearman_correlations` | ✅ DONE | Changed from union of features (with 0-padding) to intersection, preventing distortion of Spearman correlation |

### Phase 7: Documentation

| # | Location | Status | Description |
|---|---|---|---|
| ~~152~~ | `docs/todo.md` | ✅ DONE | Updated with Session 7 comprehensive audit summary (#122-152) |

