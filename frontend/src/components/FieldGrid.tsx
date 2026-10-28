import type { FieldSpec, Json, Schema, Validation, Values } from '../api';
import { FormState, visible } from '../lib/form';
import { FieldMessages, SchemaField } from './ui';

export type FormCtx = {
  schema: Schema; form: FormState; values: Values; validation: Validation | null;
  setField: (key: string, value: Json) => void; accept: (key: string) => void; dismiss: (key: string) => void; review: (key: string) => void;
};

export function messagesFor(v: Validation | null, key: string): FieldMessages {
  const pick = (list: { field: string; message: string }[] | undefined) => (list ?? []).filter((i) => i.field === key).map((i) => i.message);
  return { errors: pick(v?.errors), warnings: pick(v?.warnings), unsupported: pick(v?.unsupported) };
}

export function visibleFields(ctx: FormCtx, section: FieldSpec['section'], advanced?: boolean, exclude: string[] = []): FieldSpec[] {
  return ctx.schema.fields.filter((f) => f.section === section && (advanced === undefined || f.advanced === advanced)
    && !exclude.includes(f.key) && visible(f.key, ctx.values, f.profiles));
}

export function Field({ ctx, spec }: { ctx: FormCtx; spec: FieldSpec }): JSX.Element {
  const optionLabels = spec.key === 'model.profile' ? Object.fromEntries(Object.entries(ctx.schema.profiles).map(([k, p]) => [k, p.label])) : undefined;
  return <SchemaField spec={spec} state={ctx.form[spec.key]} messages={messagesFor(ctx.validation, spec.key)} requiredFor={ctx.validation?.required[spec.key] ?? []}
    optionLabels={optionLabels} onChange={(v) => ctx.setField(spec.key, v)} onAccept={() => ctx.accept(spec.key)} onDismiss={() => ctx.dismiss(spec.key)} onReview={() => ctx.review(spec.key)} />;
}

export function FieldGrid({ ctx, section, advanced, exclude, only }: { ctx: FormCtx; section: FieldSpec['section']; advanced?: boolean; exclude?: string[]; only?: string[] }): JSX.Element {
  let fields = visibleFields(ctx, section, advanced, exclude);
  if (only) fields = only.map((k) => ctx.schema.fields.find((f) => f.key === k)).filter((f): f is FieldSpec => !!f && visible(f.key, ctx.values, f.profiles));
  if (!fields.length) return <p className="text-xs text-[#8e899b]">No fields apply to the current selection.</p>;
  return <div className="grid gap-x-4 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">{fields.map((f) => <Field key={f.key} ctx={ctx} spec={f} />)}</div>;
}
