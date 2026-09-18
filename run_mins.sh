#!/bin/bash
cd /home/ubuntu/polymarket
for mn in 0.98 0.985 0.99 0.995; do
  echo "##### MIN_ENTRY=$mn"
  python3 sim_pnl.py 1.0 $mn | grep -A4 "YES side"
  python3 slippage.py 1.0 $mn
done
