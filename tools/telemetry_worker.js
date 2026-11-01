// Optional anonymous activity collector for the wowws-bot update checker.
// Deploy on Cloudflare Workers with a KV namespace bound as TELEMETRY.
// Only random install IDs and version/platform strings are stored; no account,
// battle or ship data is sent.

const DAY = 24 * 60 * 60 * 1000;

function json(payload, status = 200, headers = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      "access-control-allow-origin": "*",
      ...headers,
    },
  });
}

async function stats(env) {
  const cached = await env.TELEMETRY.get("stats:cache", { type: "json" });
  if (cached && Number(cached.generated_at) > Date.now() - 5 * 60 * 1000) {
    return cached;
  }
  const now = Date.now();
  let cursor;
  let total = 0;
  let active7 = 0;
  let active30 = 0;
  do {
    const page = await env.TELEMETRY.list({ prefix: "id:", cursor });
    for (const key of page.keys) {
      total += 1;
      const lastSeen = Number(key.metadata?.last_seen || 0);
      if (lastSeen && now - lastSeen <= 7 * DAY) active7 += 1;
      if (lastSeen && now - lastSeen <= 30 * DAY) active30 += 1;
    }
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  const result = {
    active_7d: active7,
    active_30d: active30,
    total,
    generated_at: now,
  };
  await env.TELEMETRY.put("stats:cache", JSON.stringify(result), {
    expirationTtl: 300,
  });
  return result;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET, POST, OPTIONS",
          "access-control-allow-headers": "content-type",
        },
      });
    }
    if (request.method === "POST" && url.pathname === "/heartbeat") {
      let body;
      try {
        body = await request.json();
      } catch (error) {
        return json({ error: "invalid json" }, 400);
      }
      const id = String(body?.install_id || "").toLowerCase();
      if (!/^[0-9a-f]{32}$/.test(id)) {
        return json({ error: "invalid install_id" }, 400);
      }
      const now = Date.now();
      let previous = null;
      try {
        previous = await env.TELEMETRY.get(`id:${id}`, { type: "json" });
      } catch (error) {
        previous = null;
      }
      const record = {
        first_seen: Number(previous?.first_seen) || now,
        last_seen: now,
        version: String(body?.version || "").slice(0, 40),
        platform: String(body?.platform || "").slice(0, 40),
      };
      await env.TELEMETRY.put(`id:${id}`, JSON.stringify(record), {
        expirationTtl: 400 * 24 * 60 * 60,
        metadata: { last_seen: now, version: record.version },
      });
      return json({ ok: true });
    }
    if (request.method === "GET" && url.pathname === "/stats") {
      const result = await stats(env);
      return json(result, 200, { "cache-control": "public, max-age=300" });
    }
    return json({ error: "not found" }, 404);
  },
};
