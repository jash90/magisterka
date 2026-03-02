import { useForm } from 'react-hook-form';
import type { PatientFormData } from '../schemas/patient';
import { defaultPatientValues } from '../schemas/patient';
import type { PatientInput } from '../api/types';

export function usePatientForm() {
  const form = useForm<PatientFormData>({
    defaultValues: defaultPatientValues,
  });

  function toPatientInput(data: PatientFormData): PatientInput {
    return { ...data };
  }

  return { form, toPatientInput };
}
