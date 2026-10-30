import type { FieldSpec, Json, Schema, Section, Validation, Values } from '../api';
import { FormState, lockReason, visible } from '../lib/form';
import { Accordion, FieldMessages, FixedSelect, SchemaField } from './ui';

export type FormCtx = {
  schema: Schema; form: FormState; values: Values; validation: Validation | null;
  setField: (key: string, value: Json) => void; accept: (key: string) => void; dismiss: (key: string) => void;
};

export function messagesFor(v: Validation | null, key: string): FieldMessages {
  const pick = (list: { field: string; message: string }[] | undefined) => (list ?? []).filter((i) => i.field === key).map((i) => i.message);
  return { errors: pick(v?.errors), warnings: pick(v?.warnings) };
}

export function Field({ ctx, spec }: { ctx: FormCtx; spec: FieldSpec }): JSX.Element {
  return <SchemaField spec={spec} state={ctx.form[spec.key]} messages={messagesFor(ctx.validation, spec.key)} requiredFor={ctx.validation?.required[spec.key] ?? []}
    lock={lockReason(spec.key, spec.editable, ctx.form, ctx.values)} onChange={(v) => ctx.setField(spec.key, v)} onAccept={() => ctx.accept(spec.key)} onDismiss={() => ctx.dismiss(spec.key)} />;
}

export function FieldGrid({ ctx, section, only, exclude = [], advanced = false }: { ctx: FormCtx; section?: Section; only?: string[]; exclude?: string[]; advanced?: boolean }): JSX.Element {
  const fields = only ? only.map((k) => ctx.schema.fields.find((f) => f.key === k)).filter((f): f is FieldSpec => !!f && visible(f.key, ctx.values))
    : ctx.schema.fields.filter((f) => f.section === section && f.advanced === advanced && !exclude.includes(f.key) && visible(f.key, ctx.values));
  return <div className="grid grid-cols-2 gap-x-3 gap-y-2 sm:grid-cols-3 lg:grid-cols-5">{fields.map((f) => <Field key={f.key} ctx={ctx} spec={f} />)}</div>;
}

const human = (s: string) => s.replace(/_/g, ' ');

/** Locked v1 assumptions from the feature contract; one selectable option each. */
export function AssumptionsPanel({ schema, sections }: { schema: Schema; sections: string[] }): JSX.Element {
  const groups = sections.map((s) => [s, schema.fixed_assumptions.filter((a) => a.section === s)] as const).filter(([, list]) => list.length);
  return <Accordion title="Execution assumptions" hint="locked in v1; become editable as estimation rules are added">
    <div className="space-y-3">{groups.map(([section, list]) => <div key={section}><p className="eyebrow mb-1">{human(section)}</p>
      <div className="grid grid-cols-2 gap-x-3 gap-y-2 sm:grid-cols-3 lg:grid-cols-5">{list.map((a) => <FixedSelect key={a.feature} label={human(a.feature)} value={String(a.value)} hint={a.later_extension} />)}</div></div>)}
      {sections.includes('training') && <div><p className="eyebrow mb-1">Estimator policy (illustrative, uncalibrated)</p>
        <div className="grid grid-cols-2 gap-x-3 gap-y-2 sm:grid-cols-3 lg:grid-cols-5">{Object.entries(schema.estimator_policy).filter(([k]) => k !== 'label').map(([k, v]) => <FixedSelect key={k} label={human(k)} value={String(v)} />)}</div></div>}
    </div></Accordion>;
}
