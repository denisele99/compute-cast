// Browser -> our backend. External-service API keys belong on the server.
export async function requestEstimate(payload) {
  const response = await fetch('/api/estimate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await response.json();
    throw new Error(Array.isArray(body.detail)
      ? body.detail.map(e => e.loc.at(-1) + ': ' + e.msg).join('; ')
      : body.detail || 'Estimate request failed');
  }
  return response.json();
}
