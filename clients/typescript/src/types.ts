/** Types mirrored from the VoltMem Python sidecar JSON responses. */

export type Message = {
  role: string;
  content: string;
};

/** String fact, one chat message, or a list of messages (extractable). */
export type AddData = string | Message | Message[];

/** Facet for multi-facet event storage. */
export type Facet = {
  content: string;
  domain: string;
  modality?: string;
  ttl_seconds?: number;
  expires_at?: number;
};

export type WriteResult = {
  id: string;
  memory: string;
  action: string;
  domain: string;
  detail: string;
};

export type MemoryItem = {
  id: string;
  memory: string;
  domain: string;
  source: string;
  created_at: number;
  last_confirmed_at: number;
  event_id?: string;
  modality?: string;
  expires_at?: number;
};

export type MemoryHit = MemoryItem & {
  score: number;
};

export type DomainStat = {
  prior?: number;
  inserted?: number;
  confirmed?: number;
  logged_mismatch?: number;
  audited?: number;
  audit_rate?: number;
  mismatch_rate?: number;
  [key: string]: number | undefined;
};

export type DomainStats = Record<string, DomainStat>;

/** Per-call or client-level tenant. `tenantId` wins when both are set. */
export type TenantScope = {
  /** Isolation boundary: a person or an app bucket. */
  tenantId?: string;
  /** @deprecated Use tenantId. Still accepted. */
  userId?: string;
};

export type VoltMemClientOptions = TenantScope & {
  /** Sidecar base URL, e.g. `https://voltmem.example.com` (no trailing slash required). */
  baseUrl: string;
  /** Sent as `X-API-Key` when the sidecar has `VOLTMEM_API_KEY` set. */
  apiKey?: string;
  /** Override `globalThis.fetch` (tests / custom runtimes). */
  fetch?: typeof fetch;
};

export type AddOptions = TenantScope & {
  source?: string;
  extract?: boolean;
  /** Fact kind. When set, the sidecar skips the profile classifier. */
  domain?: string;
  event_id?: string;
  modality?: string;
  expires_at?: number;
  ttl_seconds?: number;
};

export type AddEventOptions = TenantScope & {
  source?: string;
};

export type SearchOptions = TenantScope & {
  limit?: number;
  minScore?: number;
};

export type UserOptions = TenantScope;

export type MaintenanceTriggerOptions = TenantScope & {
  task?: string;
  dry_run?: boolean;
};
