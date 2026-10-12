/* The hosted client never receives infrastructure credentials. */
export class HostedClient {
  constructor(token = '', transport = fetch) { this.token = token; this.transport = (...args) => transport(...args); }
  async download(path) {
    const response = await this.transport(`/api/hosted${path}`, {headers:{Authorization:`Bearer ${this.token}`},redirect:'error'});
    if (!response.ok) { const data=await response.json().catch(()=>({})); const error=new Error(typeof data.detail==='string'?data.detail:`Download failed (${response.status}).`);error.status=response.status;throw error; }
    return response.blob();
  }
  async request(path, method = 'GET', body) {
    const headers = {Authorization: `Bearer ${this.token}`};
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    const response = await this.transport(`/api/hosted${path}`, {method, headers, body: body === undefined ? undefined : JSON.stringify(body), redirect: 'error'});
    if (response.status === 204) return null;
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}). Please try again.`);
      error.status = response.status; throw error;
    }
    return data;
  }
}
export function money(minor, currency = 'USD') {
  return new Intl.NumberFormat(undefined, {style:'currency', currency}).format(minor / 100);
}
export function expiryTime(value) { return typeof value === 'number' ? value * 1000 : Date.parse(value); }
export function approvalAllowed(quote, consent, now = Date.now()) {
  return Boolean(consent && quote && Number.isSafeInteger(quote.amount_minor) && quote.amount_minor > 0 && expiryTime(quote.expires_at) > now);
}
export function safeCheckout(url) {
  const parsed = new URL(url);
  if (parsed.protocol !== 'https:' || parsed.hostname !== 'checkout.stripe.com') throw new Error('The checkout URL was not recognized. Please contact support.');
  return parsed.href;
}
export function projectPath(projectId, jobId) {
  const project = `/projects/${encodeURIComponent(projectId)}`;
  return jobId ? `${project}/jobs/${encodeURIComponent(jobId)}` : project;
}
