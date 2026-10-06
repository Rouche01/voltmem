import type {
  AddData,
  AddOptions,
  AddEventOptions,
  ClearOptions,
  DomainStats,
  Facet,
  MaintenanceTriggerOptions,
  MemoryHit,
  MemoryItem,
  SearchOptions,
  TenantScope,
  UserOptions,
  VoltMemClientOptions,
  WriteResult,
} from "./types.js";

export type {
  AddData,
  AddOptions,
  AddEventOptions,
  ClearOptions,
  DomainStat,
  DomainStats,
  Facet,
  MaintenanceTriggerOptions,
  MemoryHit,
  MemoryItem,
  Message,
  SearchOptions,
  TenantScope,
  UserOptions,
  VoltMemClientOptions,
  WriteResult,
} from "./types.js";

export class VoltMemError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "VoltMemError";
    this.status = status;
    this.body = body;
  }
}

/**
 * Thin fetch client for the VoltMem HTTP sidecar.
 * Safe for Cloudflare Workers, Node 18+, Bun, and browsers (server-side preferred for API keys).
 */
export class VoltMemClient {
  readonly baseUrl: string;
  readonly apiKey: string | undefined;
  readonly tenantId: string | undefined;
  /** @deprecated Use tenantId. Holds the same resolved id. */
  readonly userId: string | undefined;
  private readonly fetchImpl: typeof fetch;

  constructor(options: VoltMemClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    const tenantId = options.tenantId ?? options.userId;
    this.tenantId = tenantId;
    this.userId = tenantId;
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  /** Bound client for another tenant (shares baseUrl / apiKey / fetch). */
  forTenant(tenantId: string): VoltMemClient {
    const options: VoltMemClientOptions = {
      baseUrl: this.baseUrl,
      tenantId,
      fetch: this.fetchImpl,
    };
    if (this.apiKey !== undefined) options.apiKey = this.apiKey;
    return new VoltMemClient(options);
  }

  /** @deprecated Use forTenant. */
  forUser(userId: string): VoltMemClient {
    return this.forTenant(userId);
  }

  async health(): Promise<{ status: string }> {
    return this.request<{ status: string }>("GET", "/health", { auth: false });
  }

  async add(
    data: AddData,
    options: AddOptions = {},
  ): Promise<WriteResult | WriteResult[]> {
    const tenantId = this.requireTenantId(options);
    const body: Record<string, unknown> = { data };
    if (options.source !== undefined) body.source = options.source;
    if (options.extract !== undefined) body.extract = options.extract;
    if (options.domain !== undefined && options.domain.trim()) {
      body.domain = options.domain.trim();
    }
    if (options.event_id !== undefined) body.event_id = options.event_id;
    if (options.modality !== undefined) body.modality = options.modality;
    if (options.expires_at !== undefined) body.expires_at = options.expires_at;
    if (options.ttl_seconds !== undefined) body.ttl_seconds = options.ttl_seconds;
    return this.request<WriteResult | WriteResult[]>(
      "POST",
      `/v1/tenants/${encodeURIComponent(tenantId)}/memories`,
      { body },
    );
  }

  async addEvent(
    event_id: string,
    facets: Facet[],
    options: AddEventOptions = {},
  ): Promise<WriteResult[]> {
    const tenantId = this.requireTenantId(options);
    const body: Record<string, unknown> = { event_id, facets };
    if (options.source !== undefined) body.source = options.source;
    return this.request<WriteResult[]>(
      "POST",
      `/v1/tenants/${encodeURIComponent(tenantId)}/events`,
      { body },
    );
  }

  async getEvent(
    eventId: string,
    options: UserOptions = {},
  ): Promise<MemoryItem[]> {
    const tenantId = this.requireTenantId(options);
    return this.request<MemoryItem[]>(
      "GET",
      `/v1/tenants/${encodeURIComponent(tenantId)}/events/${encodeURIComponent(eventId)}`,
    );
  }

  async search(
    query: string,
    options: SearchOptions = {},
  ): Promise<MemoryHit[]> {
    const tenantId = this.requireTenantId(options);
    const params = new URLSearchParams({ q: query });
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    if (options.minScore !== undefined) {
      params.set("min_score", String(options.minScore));
    }
    return this.request<MemoryHit[]>(
      "GET",
      `/v1/tenants/${encodeURIComponent(tenantId)}/memories/search?${params}`,
    );
  }

  async getAll(options: UserOptions = {}): Promise<MemoryItem[]> {
    const tenantId = this.requireTenantId(options);
    return this.request<MemoryItem[]>(
      "GET",
      `/v1/tenants/${encodeURIComponent(tenantId)}/memories`,
    );
  }

  async get(
    memoryId: string,
    options: UserOptions = {},
  ): Promise<MemoryItem> {
    const tenantId = this.requireTenantId(options);
    return this.request<MemoryItem>(
      "GET",
      `/v1/tenants/${encodeURIComponent(tenantId)}/memories/${encodeURIComponent(memoryId)}`,
    );
  }

  async delete(
    memoryId: string,
    options: UserOptions = {},
  ): Promise<{ deleted: boolean }> {
    const tenantId = this.requireTenantId(options);
    return this.request<{ deleted: boolean }>(
      "DELETE",
      `/v1/tenants/${encodeURIComponent(tenantId)}/memories/${encodeURIComponent(memoryId)}`,
    );
  }

  async clear(
    options: ClearOptions = {},
  ): Promise<{ cleared: boolean; domain?: string }> {
    const tenantId = this.requireTenantId(options);
    const params = new URLSearchParams();
    if (options.domain !== undefined && options.domain !== "") {
      params.set("domain", options.domain);
    }
    const qs = params.toString();
    return this.request<{ cleared: boolean; domain?: string }>(
      "DELETE",
      `/v1/tenants/${encodeURIComponent(tenantId)}/memories${qs ? `?${qs}` : ""}`,
    );
  }

  async summary(options: UserOptions = {}): Promise<Record<string, unknown>> {
    const tenantId = this.requireTenantId(options);
    return this.request<Record<string, unknown>>(
      "GET",
      `/v1/tenants/${encodeURIComponent(tenantId)}/summary`,
    );
  }

  async domainStats(options: UserOptions = {}): Promise<DomainStats> {
    const tenantId = this.requireTenantId(options);
    return this.request<DomainStats>(
      "GET",
      `/v1/tenants/${encodeURIComponent(tenantId)}/domain_stats`,
    );
  }

  async maintenanceTrigger(
    options: MaintenanceTriggerOptions = {},
  ): Promise<Record<string, unknown>> {
    const tenantId = this.requireTenantId(options);
    const body: Record<string, unknown> = {};
    if (options.task !== undefined) body.task = options.task;
    if (options.dry_run !== undefined) body.dry_run = options.dry_run;
    return this.request<Record<string, unknown>>(
      "POST",
      `/v1/tenants/${encodeURIComponent(tenantId)}/maintenance/trigger`,
      { body },
    );
  }

  async maintenanceRollback(
    runId: string,
    options: UserOptions = {},
  ): Promise<Record<string, unknown>> {
    const tenantId = this.requireTenantId(options);
    return this.request<Record<string, unknown>>(
      "POST",
      `/v1/tenants/${encodeURIComponent(tenantId)}/maintenance/rollback`,
      { body: { run_id: runId } },
    );
  }

  async maintenanceTasks(
    options: UserOptions = {},
  ): Promise<Array<{ name: string; description: string; interval: number }>> {
    const tenantId = this.requireTenantId(options);
    return this.request<Array<{ name: string; description: string; interval: number }>>(
      "GET",
      `/v1/tenants/${encodeURIComponent(tenantId)}/maintenance/tasks`,
    );
  }

  private requireTenantId(options: TenantScope): string {
    const tenantId = options.tenantId ?? options.userId ?? this.tenantId;
    if (!tenantId) {
      throw new Error(
        "tenantId is required (pass VoltMemClientOptions.tenantId or per-call tenantId)",
      );
    }
    return tenantId;
  }

  private async request<T>(
    method: string,
    path: string,
    opts: { body?: unknown; auth?: boolean } = {},
  ): Promise<T> {
    const headers: Record<string, string> = {
      Accept: "application/json",
    };
    if (opts.auth !== false && this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }

    const init: RequestInit = { method, headers };
    if (opts.body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.body);
    }

    const res = await this.fetchImpl(`${this.baseUrl}${path}`, init);

    const text = await res.text();
    let parsed: unknown = undefined;
    if (text) {
      try {
        parsed = JSON.parse(text) as unknown;
      } catch {
        parsed = text;
      }
    }

    if (!res.ok) {
      throw new VoltMemError(
        `VoltMem ${method} ${path} failed (${res.status})`,
        res.status,
        parsed,
      );
    }
    return parsed as T;
  }
}
