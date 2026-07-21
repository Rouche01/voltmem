/** Types mirrored from the VoltMem Python sidecar JSON responses. */

export type Message = {
  role: string;
  content: string;
};

/** String fact, one chat message, or a list of messages (extractable). */
export type AddData = string | Message | Message[];

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

export type VoltMemClientOptions = {
  /** Sidecar base URL, e.g. `https://voltmem.example.com` (no trailing slash required). */
  baseUrl: string;
  /** Sent as `X-API-Key` when the sidecar has `VOLTMEM_API_KEY` set. */
  apiKey?: string;
  /** Default tenant for convenience methods. */
  userId?: string;
  /** Override `globalThis.fetch` (tests / custom runtimes). */
  fetch?: typeof fetch;
};

export type AddOptions = {
  source?: string;
  extract?: boolean;
  userId?: string;
};

export type SearchOptions = {
  limit?: number;
  minScore?: number;
  userId?: string;
};

export type UserOptions = {
  userId?: string;
};
