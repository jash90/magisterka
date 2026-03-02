import { useQuery, useMutation } from '@tanstack/react-query';
import * as api from '../api/endpoints';
import type { ExplanationRequest, ChatRequest, BatchPatientInput, PatientInput } from '../api/types';

export function useDemoMode() {
  return useQuery({
    queryKey: ['demoMode'],
    queryFn: api.getDemoMode,
    retry: false,
    staleTime: 60_000,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: api.getHealth,
    retry: false,
    refetchInterval: 30_000,
  });
}

export function usePredict() {
  return useMutation({
    mutationFn: (patient: PatientInput) => api.predict(patient),
  });
}

export function useExplainShap() {
  return useMutation({
    mutationFn: (req: ExplanationRequest) => api.explainShap(req),
  });
}

export function useExplainLime() {
  return useMutation({
    mutationFn: (req: ExplanationRequest) => api.explainLime(req),
  });
}

export function useExplainComparison() {
  return useMutation({
    mutationFn: (req: ExplanationRequest) => api.explainComparison(req),
  });
}

export function useChat() {
  return useMutation({
    mutationFn: (req: ChatRequest) => api.chat(req),
  });
}

export function useBatchPredict() {
  return useMutation({
    mutationFn: (input: BatchPatientInput) => api.predictBatch(input),
  });
}
