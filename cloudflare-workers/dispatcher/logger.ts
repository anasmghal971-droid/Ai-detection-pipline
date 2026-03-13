// ============================================================
// DETECT-AI: Structured Logger
// Writes JSON logs to Supabase worker_logs table
// ============================================================

import type { Env, WorkerEvent, WorkerLog } from "../../shared/types/index";
import { createSupabaseClient } from "./supabase";

export class PipelineLogger {
  private workerId: string;
  private supabaseUrl: string;
  private supabaseKey: string;
  private buffer: WorkerLog[] = [];
  private readonly FLUSH_EVERY = 20; // batch writes

  constructor(env: Env) {
    this.workerId = env.WORKER_ID;
    this.supabaseUrl = env.SUPABASE_URL;
    this.supabaseKey = env.SUPABASE_SERVICE_KEY;
  }

  log(
    event: WorkerEvent,
    opts: {
      source_id?: string;
      sample_count?: number;
      duration_ms?: number;
      error_message?: string;
    } = {}
  ): void {
    const entry: WorkerLog = {
      event,
      worker_id: this.workerId,
      timestamp: new Date().toISOString(),
      ...opts,
    };

    // Always print to console (visible in CF dashboard)
    console.log(JSON.stringify(entry));

    this.buffer.push(entry);
    if (this.buffer.length >= this.FLUSH_EVERY) {
      this.flush(); // fire-and-forget
    }
  }

  async flush(): Promise<void> {
    if (this.buffer.length === 0) return;
    const toWrite = this.buffer.splice(0, this.buffer.length);
    try {
      const db = createSupabaseClient(this.supabaseUrl, this.supabaseKey);
      await db.from("worker_logs").insert(toWrite);
    } catch (e) {
      console.error(JSON.stringify({ event: "LOG_FLUSH_ERROR", error: String(e) }));
    }
  }
}
