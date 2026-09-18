#!/bin/bash
# Re-run the entry-cap scan with real fills + fees. Biggest cap first so tapes get cached.
cd /home/ubuntu/polymarket
for cap in 1.0 0.985 0.98 0.975 0.97 0.965; do
  echo "##### MAX_ENTRY=$cap"
  python3 sim_pnl.py $cap | grep -A4 "YES side"
  python3 slippage.py $cap
done
