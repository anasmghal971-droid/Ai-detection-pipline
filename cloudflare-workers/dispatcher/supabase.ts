// ============================================================
// DETECT-AI: Supabase Client for Cloudflare Workers
// Lightweight REST-based client (no Node.js dependencies)
// ============================================================

export interface SupabaseClient {
  from: (table: string) => QueryBuilder;
  rpc: (fn: string, params?: Record<string, unknown>) => Promise<SupabaseResponse>;
}

export interface SupabaseResponse {
  data: unknown;
  error: SupabaseError | null;
  status: number;
}

export interface SupabaseError {
  message: string;
  code: string;
  details?: string;
}

interface QueryBuilder {
  select: (columns?: string) => FilterBuilder;
  insert: (rows: unknown | unknown[], options?: { returning?: string }) => Promise<SupabaseResponse>;
  update: (values: Record<string, unknown>) => FilterBuilder;
  delete: () => FilterBuilder;
  upsert: (rows: unknown | unknown[]) => Promise<SupabaseResponse>;
}

interface FilterBuilder extends Promise<SupabaseResponse> {
  eq: (column: string, value: unknown) => FilterBuilder;
  neq: (column: string, value: unknown) => FilterBuilder;
  in: (column: string, values: unknown[]) => FilterBuilder;
  lt: (column: string, value: unknown) => FilterBuilder;
  gt: (column: string, value: unknown) => FilterBuilder;
  is: (column: string, value: null | boolean) => FilterBuilder;
  order: (column: string, opts?: { ascending?: boolean }) => FilterBuilder;
  limit: (n: number) => FilterBuilder;
  single: () => Promise<SupabaseResponse>;
}

// ── Factory Function ──────────────────────────────────────────
export function createSupabaseClient(url: string, serviceKey: string): SupabaseClient {
  const headers = {
    "Content-Type": "application/json",
    "apikey": serviceKey,
    "Authorization": `Bearer ${serviceKey}`,
    "Prefer": "return=representation",
  };

  async function executeRequest(
    path: string,
    method: string,
    body?: unknown,
    params?: Record<string, string>
  ): Promise<SupabaseResponse> {
    const fullUrl = new URL(`${url}/rest/v1/${path}`);
    if (params) {
      Object.entries(params).forEach(([k, v]) => fullUrl.searchParams.set(k, v));
    }

    const response = await fetch(fullUrl.toString(), {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    const data = response.ok ? await response.json().catch(() => null) : null;
    const error = !response.ok
      ? { message: await response.text(), code: String(response.status) }
      : null;

    return { data, error, status: response.status };
  }

  function buildFilterable(
    table: string,
    method: string,
    body?: unknown,
    baseParams: Record<string, string> = {}
  ): FilterBuilder {
    const params: Record<string, string> = { ...baseParams };
    let isSingle = false;

    const builder = {
      eq(col: string, val: unknown) {
        params[col] = `eq.${val}`;
        return builder;
      },
      neq(col: string, val: unknown) {
        params[col] = `neq.${val}`;
        return builder;
      },
      in(col: string, vals: unknown[]) {
        params[col] = `in.(${vals.join(",")})`;
        return builder;
      },
      lt(col: string, val: unknown) {
        params[col] = `lt.${val}`;
        return builder;
      },
      gt(col: string, val: unknown) {
        params[col] = `gt.${val}`;
        return builder;
      },
      is(col: string, val: null | boolean) {
        params[col] = `is.${val}`;
        return builder;
      },
      order(col: string, opts: { ascending?: boolean } = {}) {
        params["order"] = `${col}.${opts.ascending !== false ? "asc" : "desc"}`;
        return builder;
      },
      limit(n: number) {
        params["limit"] = String(n);
        return builder;
      },
      single() {
        isSingle = true;
        return executeRequest(table, method, body, params);
      },
      then(resolve: (v: SupabaseResponse) => unknown, reject?: (e: unknown) => unknown) {
        return executeRequest(table, method, body, params).then(resolve, reject);
      },
      catch(reject: (e: unknown) => unknown) {
        return executeRequest(table, method, body, params).catch(reject);
      },
      finally(fn: () => unknown) {
        return executeRequest(table, method, body, params).finally(fn);
      },
    };

    return builder as unknown as FilterBuilder;
  }

  return {
    from(table: string): QueryBuilder {
      return {
        select(columns = "*") {
          return buildFilterable(table, "GET", undefined, { select: columns }) as FilterBuilder;
        },
        async insert(rows, options = {}) {
          const hdrs = { ...headers };
          if (options.returning === "minimal") hdrs["Prefer"] = "return=minimal";
          const r = Array.isArray(rows) ? rows : [rows];
          return executeRequest(table, "POST", r);
        },
        update(values) {
          return buildFilterable(table, "PATCH", values) as FilterBuilder;
        },
        delete() {
          return buildFilterable(table, "DELETE") as FilterBuilder;
        },
        async upsert(rows) {
          const r = Array.isArray(rows) ? rows : [rows];
          const response = await fetch(`${url}/rest/v1/${table}`, {
            method: "POST",
            headers: { ...headers, "Prefer": "resolution=merge-duplicates,return=representation" },
            body: JSON.stringify(r),
          });
          const data = response.ok ? await response.json().catch(() => null) : null;
          const error = !response.ok
            ? { message: await response.text(), code: String(response.status) }
            : null;
          return { data, error, status: response.status };
        },
      };
    },

    async rpc(fn: string, params = {}) {
      return executeRequest(`rpc/${fn}`, "POST", params);
    },
  };
}
