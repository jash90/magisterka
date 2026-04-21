import { z } from 'zod';

export const patientSchema = z.object({
  wiek_rozpoznania: z.number().min(0).max(120).optional(),
  opoznienie_rozpoznia: z.number().min(0).optional(),
  manifestacja_miesno_szkiel: z.number().min(0).max(1),
  manifestacja_skora: z.number().min(0).max(1),
  manifestacja_wzrok: z.number().min(0).max(1),
  manifestacja_sercowo_naczyniowy: z.number().min(0).max(1),
  manifestacja_pokarmowy: z.number().min(0).max(1),
  manifestacja_nerki: z.number().min(0).max(1),
  manifestacja_moczowo_plciowy: z.number().min(0).max(1),
  manifestacja_zajecie_csn: z.number().min(0).max(1),
  manifestacja_neurologiczny: z.number().min(0).max(1),
  liczba_zajetych_narzadow: z.number().min(0).max(20),
  zaostrz_wymagajace_hospital: z.number().min(0).max(1),
  zaostrz_wymagajace_oit: z.number().min(0).max(1),
  kreatynina: z.number().min(0).optional(),
  eozynofilia_krwi_obwodowej_wartosc: z.number().min(0).optional(),
  pulsy: z.number().min(0).max(1),
  czas_sterydow: z.number().min(0).optional(),
  plazmaferezy: z.number().min(0).max(1),
  biopsja_wynik: z.number().min(0).max(1),
});

export type PatientFormData = z.infer<typeof patientSchema>;

export const defaultPatientValues: PatientFormData = {
  wiek_rozpoznania: 50,
  opoznienie_rozpoznia: 6,
  manifestacja_miesno_szkiel: 0,
  manifestacja_skora: 0,
  manifestacja_wzrok: 0,
  manifestacja_sercowo_naczyniowy: 0,
  manifestacja_nerki: 0,
  manifestacja_pokarmowy: 0,
  manifestacja_moczowo_plciowy: 0,
  manifestacja_zajecie_csn: 0,
  manifestacja_neurologiczny: 0,
  liczba_zajetych_narzadow: 2,
  zaostrz_wymagajace_hospital: 0,
  zaostrz_wymagajace_oit: 0,
  kreatynina: 100,
  eozynofilia_krwi_obwodowej_wartosc: 0,
  pulsy: 0,
  czas_sterydow: 12,
  plazmaferezy: 0,
  biopsja_wynik: 0,
};
