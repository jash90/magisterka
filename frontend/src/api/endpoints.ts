import apiClient from './client';
import type {
  PatientInput,
  PredictionOutput,
  ExplanationRequest,
  SHAPExplanation,
  LIMEExplanation,
  ComparisonResult,
  ChatRequest,
  ChatResponse,
  BatchPatientInput,
  BatchPredictionOutput,
  DemoModeStatus,
  HealthCheckResponse,
  AgentConversationResponse,
  MultiModelPredictionOutput,
} from './types';

export async function predict(patient: PatientInput): Promise<PredictionOutput> {
  const { data } = await apiClient.post<PredictionOutput>('/predict', patient);
  return data;
}

export async function predictAll(patient: PatientInput): Promise<MultiModelPredictionOutput> {
  const { data } = await apiClient.post<MultiModelPredictionOutput>('/predict/all', patient);
  return data;
}

export async function predictBatch(input: BatchPatientInput): Promise<BatchPredictionOutput> {
  const { data } = await apiClient.post<BatchPredictionOutput>('/predict/batch', input, {
    timeout: 300000,
  });
  return data;
}

export async function explainShap(req: ExplanationRequest): Promise<SHAPExplanation> {
  const { data } = await apiClient.post<SHAPExplanation>('/explain/shap', req);
  return data;
}

export async function explainLime(req: ExplanationRequest): Promise<LIMEExplanation> {
  const { data } = await apiClient.post<LIMEExplanation>('/explain/lime', req);
  return data;
}

export async function explainComparison(req: ExplanationRequest): Promise<ComparisonResult> {
  const { data } = await apiClient.post<ComparisonResult>('/explain/comparison', req);
  return data;
}

export async function chat(req: ChatRequest): Promise<ChatResponse> {
  const { data } = await apiClient.post<ChatResponse>('/chat', req);
  return data;
}

export async function getDemoMode(): Promise<DemoModeStatus> {
  const { data } = await apiClient.get<DemoModeStatus>('/config/demo-mode');
  return data;
}

export async function getHealth(): Promise<HealthCheckResponse> {
  const { data } = await apiClient.get<HealthCheckResponse>('/health');
  return data;
}

export interface AgentChatPayload {
  message: string;
  conversation_history: Array<{ role: string; content: string }>;
  collected_data: Record<string, number | string>;
  current_step: number;
  phase: 'collecting' | 'prediction' | 'discussion';
}

export async function agentChat(req: AgentChatPayload): Promise<AgentConversationResponse> {
  const { data } = await apiClient.post<AgentConversationResponse>('/agent/chat', req);
  return data;
}
