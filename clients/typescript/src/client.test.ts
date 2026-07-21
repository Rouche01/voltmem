import assert from "node:assert/strict";
import { mock, test } from "node:test";

import { VoltMemClient, VoltMemError } from "./index.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("add / search / domainStats hit expected paths and headers", async () => {
  const calls: Array<{ url: string; init: RequestInit | undefined }> = [];
  const fetchMock = mock.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    if (url.endsWith("/memories") && init?.method === "POST") {
      return jsonResponse(200, {
        id: "m1",
        memory: "I prefer darker colors",
        action: "inserted",
        domain: "style_preference",
        detail: "",
      });
    }
    if (url.includes("/memories/search")) {
      return jsonResponse(200, [
        {
          id: "m1",
          memory: "I prefer darker colors",
          domain: "style_preference",
          source: "explicit_statement",
          created_at: 1,
          last_confirmed_at: 1,
          score: 0.9,
        },
      ]);
    }
    if (url.endsWith("/domain_stats")) {
      return jsonResponse(200, {
        style_preference: { prior: 0.08, inserted: 1, audit_rate: 0 },
      });
    }
    return jsonResponse(404, { detail: "unexpected" });
  });

  const client = new VoltMemClient({
    baseUrl: "https://voltmem.example.com/",
    apiKey: "secret",
    userId: "alice",
    fetch: fetchMock as unknown as typeof fetch,
  });

  const added = await client.add("I prefer darker colors");
  assert.equal((added as { id: string }).id, "m1");

  const hits = await client.search("style preferences", { limit: 3 });
  assert.equal(hits[0]?.domain, "style_preference");

  const stats = await client.domainStats();
  assert.equal(stats.style_preference?.prior, 0.08);

  assert.equal(calls.length, 3);
  assert.match(calls[0]!.url, /\/v1\/users\/alice\/memories$/);
  assert.equal(calls[0]!.init?.method, "POST");
  assert.equal(
    (calls[0]!.init?.headers as Record<string, string>)["X-API-Key"],
    "secret",
  );
  assert.match(calls[1]!.url, /q=style\+preferences/);
  assert.match(calls[1]!.url, /limit=3/);
  assert.match(calls[2]!.url, /\/domain_stats$/);
});

test("health skips API key", async () => {
  const fetchMock = mock.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    const headers = init?.headers as Record<string, string>;
    assert.equal(headers["X-API-Key"], undefined);
    return jsonResponse(200, { status: "ok" });
  });

  const client = new VoltMemClient({
    baseUrl: "https://voltmem.example.com",
    apiKey: "secret",
    fetch: fetchMock as unknown as typeof fetch,
  });
  const health = await client.health();
  assert.deepEqual(health, { status: "ok" });
});

test("throws VoltMemError on non-2xx", async () => {
  const client = new VoltMemClient({
    baseUrl: "https://voltmem.example.com",
    userId: "alice",
    fetch: (async () =>
      jsonResponse(401, { detail: "invalid or missing X-API-Key" })) as typeof fetch,
  });

  await assert.rejects(
    () => client.getAll(),
    (err: unknown) => {
      assert.ok(err instanceof VoltMemError);
      assert.equal((err as VoltMemError).status, 401);
      return true;
    },
  );
});

test("forUser overrides tenant", async () => {
  let seen = "";
  const client = new VoltMemClient({
    baseUrl: "https://voltmem.example.com",
    userId: "alice",
    fetch: (async (input) => {
      seen = String(input);
      return jsonResponse(200, []);
    }) as typeof fetch,
  });

  await client.forUser("bob").getAll();
  assert.match(seen, /\/v1\/users\/bob\/memories$/);
});

test("requires userId", async () => {
  const client = new VoltMemClient({
    baseUrl: "https://voltmem.example.com",
    fetch: (async () => jsonResponse(200, [])) as typeof fetch,
  });
  await assert.rejects(() => client.getAll(), /userId is required/);
});
