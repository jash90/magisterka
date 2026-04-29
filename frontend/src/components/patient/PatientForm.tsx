import { useState } from 'react';
import { usePatientForm } from '../../hooks/usePatientForm';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';

interface PatientFormProps {
  onSubmit: (patient: PatientInput) => void;
  isSubmitting: boolean;
}

/* ---------- Section (collapsible) ---------- */
function Section({ title, defaultOpen = false, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-gray-700/60 bg-gray-800/30">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-semibold text-blue-300 transition hover:bg-gray-800/50"
      >
        {title}
        <span className="text-xs text-gray-500">{open ? 'zwiń' : 'rozwiń'}</span>
      </button>
      {open && <div className="space-y-4 px-4 pb-4">{children}</div>}
    </div>
  );
}

/* ---------- Number input ---------- */
function NumberField({ label, register, name, min, max, step, unit }: {
  label: string;
  register: ReturnType<typeof usePatientForm>['form']['register'];
  name: string;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium text-gray-400">{label}</span>
      <div className="relative">
        <input
          type="number"
          {...register(name as never, { valueAsNumber: true })}
          min={min}
          max={max}
          step={step ?? 1}
          className="w-full rounded-lg border border-gray-600/80 bg-gray-700/60 px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500/30"
        />
        {unit && (
          <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500">
            {unit}
          </span>
        )}
      </div>
    </label>
  );
}

/* ---------- Checkbox with toggle style ---------- */
function ToggleField({ label, checked, onChange }: {
  label: string;
  checked: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between rounded-lg border border-gray-700/40 bg-gray-800/20 px-3 py-2 transition hover:bg-gray-800/40">
      <span className="text-sm text-gray-300">{label}</span>
      <div className="relative">
        <input
          type="checkbox"
          checked={checked}
          onChange={onChange}
          className="peer sr-only"
        />
        <div className="h-5 w-9 rounded-full bg-gray-600 transition peer-checked:bg-blue-600" />
        <div className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition peer-checked:translate-x-4" />
      </div>
    </label>
  );
}

/* ---------- Main form ---------- */
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
      className="space-y-4"
    >
      <h2 className="text-xl font-bold text-blue-300">{pl.sidebar.patientData}</h2>

      {/* --- Demographics --- */}
      <Section title={t.sections.demographics} defaultOpen>
        <div className="grid grid-cols-2 gap-4">
          <NumberField label={t.fields.wiek_rozpoznania} register={register} name="wiek_rozpoznania" min={0} max={120} unit="lat" />
          <NumberField label={t.fields.opoznienie_rozpoznia} register={register} name="opoznienie_rozpoznia" min={0} unit="mies." />
        </div>
      </Section>

      {/* --- Organ manifestations --- */}
      <Section title={t.sections.organs} defaultOpen>
        <div className="block">
          <span className="mb-2 block text-xs font-medium text-gray-400">{t.fields.liczba_narzadow}</span>
          <div className="flex items-center gap-3">
            <span className="w-4 text-right text-xs text-gray-500">0</span>
            <input
              type="range"
              {...register('liczba_zajetych_narzadow', { valueAsNumber: true })}
              min={0}
              max={10}
              className="flex-1 accent-blue-500"
            />
            <span className="w-4 text-xs text-gray-500">10</span>
            <span className="ml-2 min-w-[2rem] rounded-md bg-gray-700 px-2 py-0.5 text-center text-sm font-bold text-white">
              {watch('liczba_zajetych_narzadow')}
            </span>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {([
            ['manifestacja_nerki', t.fields.nerki],
            ['manifestacja_sercowo_naczyniowy', t.fields.serce],
            ['manifestacja_zajecie_csn', t.fields.csn],
            ['manifestacja_neurologiczny', t.fields.neuro],
            ['manifestacja_pokarmowy', t.fields.pokarmowy],
            ['manifestacja_miesno_szkiel', t.fields.miesno_szkiel],
            ['manifestacja_skora', t.fields.skora],
            ['manifestacja_wzrok', t.fields.wzrok],
            ['manifestacja_moczowo_plciowy', t.fields.moczowo_plciowy],
          ] as const).map(([name, label]) => (
            <ToggleField
              key={name}
              label={label}
              checked={watch(name as never) === 1}
              onChange={handleCheck(name)}
            />
          ))}
        </div>
      </Section>

      {/* --- Disease course --- */}
      <Section title={t.sections.course} defaultOpen>
        <div className="grid grid-cols-2 gap-2">
          <ToggleField
            label={t.fields.hospital}
            checked={watch('zaostrz_wymagajace_hospital') === 1}
            onChange={handleCheck('zaostrz_wymagajace_hospital')}
          />
          <ToggleField
            label={t.fields.oit}
            checked={watch('zaostrz_wymagajace_oit') === 1}
            onChange={handleCheck('zaostrz_wymagajace_oit')}
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <NumberField label={t.fields.kreatynina} register={register} name="kreatynina" min={0} step={0.1} unit="μmol/L" />
          <NumberField label={t.fields.eozynofilia} register={register} name="eozynofilia_krwi_obwodowej_wartosc" min={0} step={0.1} />
        </div>
      </Section>

      {/* --- Treatment --- */}
      <Section title={t.sections.treatment} defaultOpen>
        <div className="grid grid-cols-2 gap-2">
          <ToggleField
            label={t.fields.pulsy}
            checked={watch('pulsy') === 1}
            onChange={handleCheck('pulsy')}
          />
          <ToggleField
            label={t.fields.plazmaferezy}
            checked={watch('plazmaferezy') === 1}
            onChange={handleCheck('plazmaferezy')}
          />
        </div>
        <NumberField label={t.fields.czas_sterydow} register={register} name="czas_sterydow" min={0} unit="mies." />
      </Section>

      {/* --- Diagnostics --- */}
      <Section title={t.sections.diagnostics} defaultOpen>
        <ToggleField
          label={t.fields.biopsja}
          checked={watch('biopsja_wynik') === 1}
          onChange={handleCheck('biopsja_wynik')}
        />
      </Section>

      {/* --- Submit --- */}
      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full rounded-xl bg-blue-600 py-3 text-sm font-bold text-white shadow-lg shadow-blue-600/20 transition hover:bg-blue-700 hover:shadow-blue-600/30 disabled:opacity-50"
      >
        {isSubmitting ? 'Analizuję...' : pl.sidebar.analyze}
      </button>
    </form>
  );
}
