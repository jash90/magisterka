import { useState } from 'react';
import { usePatientForm } from '../../hooks/usePatientForm';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';

interface PatientFormProps {
  onSubmit: (patient: PatientInput) => void;
  isSubmitting: boolean;
}

function Section({ title, defaultOpen = false, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-3 py-2.5 text-sm font-semibold text-blue-300"
      >
        {title}
        <span className="text-gray-400">{open ? '−' : '+'}</span>
      </button>
      {open && <div className="space-y-3 px-3 pb-3">{children}</div>}
    </div>
  );
}

function NumberField({ label, register, name, min, max, step }: {
  label: string;
  register: ReturnType<typeof usePatientForm>['form']['register'];
  name: string;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs text-gray-400">{label}</span>
      <input
        type="number"
        {...register(name as never, { valueAsNumber: true })}
        min={min}
        max={max}
        step={step ?? 1}
        className="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-1.5 text-sm text-white focus:border-blue-500 focus:outline-none"
      />
    </label>
  );
}


export function PatientForm({ onSubmit, isSubmitting }: PatientFormProps) {
  const { form, toPatientInput } = usePatientForm();
  const { register, handleSubmit, setValue, watch } = form;
  const t = pl.form;

  const handleCheck = (name: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setValue(name as never, (e.target.checked ? 1 : 0) as never);
  };

  return (
    <form
      onSubmit={handleSubmit((data) => onSubmit(toPatientInput(data)))}
      className="space-y-3"
    >
      <h2 className="text-lg font-bold text-blue-300">{pl.sidebar.patientData}</h2>

      <Section title={t.sections.demographics} defaultOpen>
        <NumberField label={t.fields.wiek_rozpoznania} register={register} name="wiek_rozpoznania" min={0} max={120} />
        <NumberField label={t.fields.opoznienie_rozpoznia} register={register} name="opoznienie_rozpoznia" min={0} />
      </Section>

      <Section title={t.sections.organs} defaultOpen>
        <label className="block">
          <span className="mb-1 block text-xs text-gray-400">{t.fields.liczba_narzadow}</span>
          <input
            type="range"
            {...register('liczba_zajetych_narzadow', { valueAsNumber: true })}
            min={0}
            max={10}
            className="w-full"
          />
          <span className="text-xs text-gray-400">{watch('liczba_zajetych_narzadow')}</span>
        </label>
        {[
          ['manifestacja_nerki', t.fields.nerki],
          ['manifestacja_sercowo_naczyniowy', t.fields.serce],
          ['manifestacja_zajecie_csn', t.fields.csn],
          ['manifestacja_neurologiczny', t.fields.neuro],
          ['manifestacja_pokarmowy', t.fields.pokarmowy],
          ['manifestacja_miesno_szkiel', t.fields.miesno_szkiel],
          ['manifestacja_skora', t.fields.skora],
          ['manifestacja_wzrok', t.fields.wzrok],
          ['manifestacja_moczowo_plciowy', t.fields.moczowo_plciowy],
        ].map(([name, label]) => (
          <label key={name} className="flex items-center gap-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={watch(name as never) === 1}
              onChange={handleCheck(name)}
              className="rounded border-gray-600 bg-gray-700"
            />
            {label}
          </label>
        ))}
      </Section>

      <Section title={t.sections.course}>
        <label className="flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={watch('zaostrz_wymagajace_hospital') === 1}
            onChange={handleCheck('zaostrz_wymagajace_hospital')}
            className="rounded border-gray-600 bg-gray-700"
          />
          {t.fields.hospital}
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={watch('zaostrz_wymagajace_oit') === 1}
            onChange={handleCheck('zaostrz_wymagajace_oit')}
            className="rounded border-gray-600 bg-gray-700"
          />
          {t.fields.oit}
        </label>
        <NumberField label={t.fields.kreatynina} register={register} name="kreatynina" min={0} step={0.1} />
        <NumberField label={t.fields.eozynofilia} register={register} name="eozynofilia_krwi_obwodowej_wartosc" min={0} step={0.1} />
      </Section>

      <Section title={t.sections.treatment}>
        <label className="flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={watch('pulsy') === 1}
            onChange={handleCheck('pulsy')}
            className="rounded border-gray-600 bg-gray-700"
          />
          {t.fields.pulsy}
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={watch('plazmaferezy') === 1}
            onChange={handleCheck('plazmaferezy')}
            className="rounded border-gray-600 bg-gray-700"
          />
          {t.fields.plazmaferezy}
        </label>
        <NumberField label={t.fields.czas_sterydow} register={register} name="czas_sterydow" min={0} />
      </Section>

      <Section title={t.sections.diagnostics}>
        <label className="flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={watch('biopsja_wynik') === 1}
            onChange={handleCheck('biopsja_wynik')}
            className="rounded border-gray-600 bg-gray-700"
          />
          {t.fields.biopsja}
        </label>
      </Section>

      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full rounded-lg bg-blue-600 py-2.5 text-sm font-bold text-white transition hover:bg-blue-700 disabled:opacity-50"
      >
        {isSubmitting ? 'Analizuję...' : pl.sidebar.analyze}
      </button>
    </form>
  );
}
