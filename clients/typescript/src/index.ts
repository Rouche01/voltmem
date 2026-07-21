import type {
  AddData,
  AddOptions,
  DomainStats,
  MemoryHit,
  MemoryItem,
  SearchOptions,
  UserOptions,
  VoltMemClientOptions,
  WriteResult,
} from "./types.js";

export type {
  AddData,
  AddOptions,
  DomainStat,
  DomainStats,
  MemoryHit,
  MemoryItem,
  Message,
  SearchOptions,
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
  readonly userId: string | undefined;
  private readonly fetchImpl: typeof fetch;

  constructor(options: VoltMemClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.userId = options.userId;
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  /** Bound client for another tenant (shares baseUrl / apiKey / fetch). */
  forUser(userId: string): VoltMemClient {
    const options: VoltMemClientOptions = {
      baseUrl: this.baseUrl,
      userId,
      fetch: this.fetchImpl,
    };
    if (this.apiKey !== undefined) options.apiKey = this.apiKey;
    return new VoltMemClient(options);
  }

  async health(): Promise<{ status: string }> {
    return this.request<{ status: string }>("GET", "/health", { auth: false });
  }

  async add(
    data: AddData,
    options: AddOptions = {},
  ): Promise<WriteResult | WriteResult[]> {
    const userId = this.requireUserId(options.userId);
    const body: Record<string, unknown> = { data };
    if (options.source !== undefined) body.source = options.source;
    if (options.extract !== undefined) body.extract = options.extract;
    return this.request<WriteResult | WriteResult[]>(
      "POST",
      `/v1/users/${encodeURIComponent(userId)}/memories`,
      { body },
    );
  }

  async search(
    query: string,
    options: SearchOptions = {},
  ): Promise<MemoryHit[]> {
    const userId = this.requireUserId(options.userId);
    const params = new URLSearchParams({ q: query });
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    if (options.minScore !== undefined) {
      params.set("min_score", String(options.minScore));
    }
    return this.request<MemoryHit[]>(
      "GET",
      `/v1/users/${encodeURIComponent(userId)}/memories/search?${params}`,
    );
  }

  async getAll(options: UserOptions = {}): Promise<MemoryItem[]> {
    const userId = this.requireUserId(options.userId);
    return this.request<MemoryItem[]>(
      "GET",
      `/v1/users/${encodeURIComponent(userId)}/memories`,
    );
  }

  async get(
    memoryId: string,
    options: UserOptions = {},
  ): Promise<MemoryItem> {
    const userId = this.requireUserId(options.userId);
    return this.request<MemoryItem>(
      "GET",
      `/v1/users/${encodeURIComponent(userId)}/memories/${encodeURIComponent(memoryId)}`,
    );
  }

  async delete(
    memoryId: string,
    options: UserOptions = {},
  ): Promise<{ deleted: boolean }> {
    const userId = this.requireUserId(options.userId);
    return this.request<{ deleted: boolean }>(
      "DELETE",
      `/v1/users/${encodeURIComponent(userId)}/memories/${encodeURIComponent(memoryId)}`,
    );
  }

  async clear(options: UserOptions = {}): Promise<{ cleared: boolean }> {
    const userId = this.requireUserId(options.userId);
    return this.request<{ cleared: boolean }>(
      "DELETE",
      `/v1/users/${encodeURIComponent(userId)}/memories`,
    );
  }

  async summary(options: UserOptions = {}): Promise<Record<string, unknown>> {
    const userId = this.requireUserId(options.userId);
    return this.request<Record<string, unknown>>(
      "GET",
      `/v1/users/${encodeURIComponent(userId)}/summary`,
    );
  }

  async domainStats(options: UserOptions = {}): Promise<DomainStats> {
    const userId = this.requireUserId(options.userId);
    return this.request<DomainStats>(
      "GET",
      `/v1/users/${encodeURIComponent(userId)}/domain_stats`,
    );
  }

  private requireUserId(override?: string): string {
    const userId = override ?? this.userId;
    if (!userId) {
      throw new Error(
        "userId is required (pass VoltMemClientOptions.userId or per-call userId)",
      );
    }
    return userId;
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
