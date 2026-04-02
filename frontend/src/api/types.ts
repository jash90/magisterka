// Mirror of schemas.py — 20 features matching XGBoost model

export type RiskLevel = 'low' | 'moderate' | 'high';
export type XAIMethod = 'lime' | 'shap' | 'dalex' | 'ebm';
export type HealthLiteracyLevel = 'basic' | 'advanced' | 'clinician';

export interface PatientInput {
  wiek_rozpoznania?: number;
  opoznienie_rozpoznia?: number;
  manifestacja_miesno_szkiel: number;
  manifestacja_skora: number;
  manifestacja_wzrok: number;
  manifestacja_sercowo_naczyniowy: number;
  manifestacja_pokarmowy: number;
  manifestacja_nerki: number;
  manifestacja_moczowo_plciowy: number;
  manifestacja_zajecie_csn: number;
  manifestacja_neurologiczny: number;
  liczba_zajetych_narzadow: number;
  zaostrz_wymagajace_hospital: number;
  zaostrz_wymagajace_oit: number;
  kreatynina?: number;
  eozynofilia_krwi_obwodowej_wartosc?: number;
  pulsy: number;
  czas_sterydow?: number;
  plazmaferezy: number;
  biopsja_wynik: number;
}

export interface PredictionOutput {
  probability: number;
  risk_level: RiskLevel;
  prediction: number;
  confidence_interval?: { lower: number; upper: number };
}

export interface ModelPrediction {
  model_name: string;
  probability: number;
  risk_level: RiskLevel;
  prediction: number;
}

export interface MultiModelPredictionOutput {
  models: ModelPrediction[];
  ensemble_probability: number;
  ensemble_risk_level: RiskLevel;
  primary_model: string;
}

export interface FeatureContribution {
  feature: string;
  value: number;
  contribution: number;
  direction: string;
}

export interface SHAPExplanation {
  method: string;
  base_value: number;
  shap_values: Record<string, number>;
  feature_contributions: FeatureContribution[];
  risk_factors: FeatureContribution[];
  protective_factors: FeatureContribution[];
  prediction: PredictionOutput;
}

export interface LIMEExplanation {
  method: string;
  intercept: number;
  feature_weights: Array<Record<string, unknown>>;
  risk_factors: Array<Record<string, unknown>>;
  protective_factors: Array<Record<string, unknown>>;
  local_prediction: number;
  prediction: PredictionOutput;
}

export interface ComparisonResult {
  methods_compared: string[];
  ranking_agreement: number;
  common_top_features: string[];
  individual_rankings: Record<string, string[]>;
  spearman_correlations: Record<string, number>;
}

export interface DALEXExplanation {
  method: 'DALEX';
  intercept: number;
  prediction: number;
  risk_factors: { feature: string; contribution: number }[];
  protective_factors: { feature: string; contribution: number }[];
  variable_importance?: Record<string, number>;
}

export interface EBMExplanation {
  method: 'EBM';
  prediction: number;
  probability: number;
  risk_level: RiskLevel;
  global_importance: Record<string, number>;
  local_contributions: { feature: string; contribution: number }[];
  interactions: string[];
}

export interface ChatRequest {
  message: string;
  patient: PatientInput;
  health_literacy: HealthLiteracyLevel;
  conversation_history: Array<{ role: string; content: string }>;
}

export interface ChatPredictionData {
  prediction: PredictionOutput;
  factors: Array<{ feature: string; contribution: number; direction: string }>;
  base_value: number;
}

export interface ChatResponse {
  response: string;
  detected_concerns?: string[];
  follow_up_suggestions?: string[];
  prediction_data?: ChatPredictionData | null;
}

export interface AgentConversationResponse {
  response: string;
  collected_data: Record<string, number | string>;
  current_step: number;
  phase: 'collecting' | 'prediction' | 'discussion';
  missing_fields: string[];
  prediction_data?: ChatPredictionData | null;
  follow_up_suggestions: string[];
  field_meta?: {
    field: string;
    type: 'number' | 'boolean';
    widget: 'slider' | 'buttons' | 'input';
    min?: number;
    max?: number;
    step?: number;
    unit?: string;
    options?: string[];
  } | null;
}

export interface ExplanationRequest {
  patient: PatientInput;
  method: XAIMethod;
  num_features: number;
}

export interface BatchPatientInput {
  patients: PatientInput[];
  include_risk_factors: boolean;
  top_n_factors: number;
}

export interface RiskFactorItem {
  feature: string;
  value: number;
  importance: number;
  direction: string;
}

export interface BatchPatientResult {
  patient_id?: string;
  index: number;
  prediction: PredictionOutput;
  top_risk_factors?: RiskFactorItem[];
  processing_status: string;
  error_message?: string;
}

export interface BatchSummary {
  total_count: number;
  low_risk_count: number;
  moderate_risk_count: number;
  high_risk_count: number;
  avg_probability: number;
  median_probability: number;
  min_probability: number;
  max_probability: number;
}

export interface BatchPredictionOutput {
  total_patients: number;
  processed_count: number;
  success_count: number;
  error_count: number;
  processing_time_ms: number;
  mode: string;
  summary: BatchSummary;
  results: BatchPatientResult[];
  errors: Array<{
    patient_index: number;
    patient_id?: string;
    error_type: string;
    error_message: string;
    is_recoverable: boolean;
  }>;
}

export interface DemoModeStatus {
  demo_allowed: boolean;
  model_loaded: boolean;
  current_mode: string;
  force_api_mode: boolean;
}

export interface HealthCheckResponse {
  status: string;
  model_loaded: boolean;
  api_version: string;
  timestamp: string;
}

export interface GlobalImportance {
  feature_importance: Record<string, number>;
  top_features: string[];
  method: string;
  n_samples: number;
}

// Batch results row for display table
export interface BatchResultRow {
  patient_id: string;
  wiek_rozpoznania: number;
  liczba_narzadow: number;
  probability: number;
  probability_pct: string;
  risk_level: RiskLevel;
  risk_level_pl: string;
  prediction: number;
  top_factors: string;
  processing_mode: string;
}
