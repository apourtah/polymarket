#!/bin/bash
cd /home/ubuntu/polymarket
for cut in 48 24 12 3 0 -3 -6 -12; do
  echo "##### CUTOFF_HRS=$cut"
  python3 sim_pnl.py 1.0 0.99 $cut | grep -A4 "YES side"
  python3 slippage.py 1.0 0.99 $cut | grep -E "fill status|market-order|P&L at real"
done
