import { useState } from "react";
import { Offer } from "../workflowApi";

export function HardwareComparison({
  offers,
  onSelect,
}: {
  offers: Offer[];
  onSelect: (offer: Offer) => void;
}) {
  const [query, setQuery] = useState("");
  const [region, setRegion] = useState("all");
  const [fitsOnly, setFitsOnly] = useState(false);
  const [selectedId, setSelectedId] = useState("");
  const visible = offers.filter(
    (offer) =>
      (region === "all" || offer.region === region) &&
      (!fitsOnly || offer.fits === true) &&
      `${offer.name} ${offer.platform}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const selected =
    visible.find((offer) => offer.id === selectedId) ?? visible[0];
  return (
    <section className="hardware-comparison">
      <p className="eyebrow">EXPLORE OPTIONS</p>
      <h2>Hardware comparison</h2>
      <div className="comparison-filters">
        <label className="input-label">
          Search comparison
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="GPU name…"
          />
        </label>
        <label className="input-label">
          Comparison region
          <select
            aria-label="Comparison region"
            value={region}
            onChange={(event) => setRegion(event.target.value)}
          >
            <option value="all">All regions</option>
            {Array.from(new Set(offers.map((offer) => offer.region))).map(
              (value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ),
            )}
          </select>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={fitsOnly}
            onChange={(event) => setFitsOnly(event.target.checked)}
          />
          Within memory limit only
        </label>
      </div>
      <div className="comparison-layout">
        <div
          className="comparison-list"
          role="region"
          aria-label="GPU comparison options"
          tabIndex={0}
        >
          {visible.length ? (
            visible.map((offer) => (
              <button
                className="comparison-option"
                key={offer.id}
                aria-pressed={selected?.id === offer.id}
                onClick={() => setSelectedId(offer.id)}
              >
                <span>{offer.name.replace("Preemptible ", "")}</span>
                <small>
                  {offer.region} ·{" "}
                  {offer.vram_gib === null
                    ? "Memory unknown"
                    : `${offer.vram_gib.toFixed(1)} GiB`}{" "}
                  · ${offer.price_per_gpu_hour}/GPU-h
                </small>
              </button>
            ))
          ) : (
            <p className="muted">
              No matching GPUs. Try another region or clear your filters.
            </p>
          )}
        </div>
        {selected && (
          <div className="comparison-detail" aria-live="polite">
            <p className="eyebrow">{selected.region}</p>
            <h3>{selected.name.replace("Preemptible ", "")}</h3>
            <dl>
              <div>
                <dt>GPU memory</dt>
                <dd>
                  {selected.vram_gib === null
                    ? "Unknown"
                    : `${selected.vram_gib.toFixed(1)} GiB`}
                </dd>
              </div>
              <div>
                <dt>Spot price</dt>
                <dd>${selected.price_per_gpu_hour}/GPU-hour</dd>
              </div>
              <div>
                <dt>Memory fit</dt>
                <dd>
                  {selected.fits === null
                    ? "Needs more metadata"
                    : selected.fits
                      ? "Within limit"
                      : "Exceeds limit"}
                </dd>
              </div>
            </dl>
            <button className="secondary" onClick={() => onSelect(selected)}>
              Use this GPU
            </button>
          </div>
        )}
      </div>
      <p className="muted comparison-note">
        Fit uses 90% of each card’s memory. Prices are saved spot quotes;
        confirm runtime on the selected GPU.
      </p>
    </section>
  );
}
