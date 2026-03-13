"""
One-shot labeler for GitHub Actions cron.
Runs a single batch and exits (Actions handles the scheduling).
"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from ensemble_labeler import label_batch, record_metrics, check_shard_trigger, SupabaseClient
import aiohttp

async def main():
    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        db = SupabaseClient(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"], session)
        labeled = await label_batch(db, session)
        if labeled > 0:
            await check_shard_trigger(db)
            await record_metrics(db)
        print(f"Labeled {labeled} samples")

asyncio.run(main())
