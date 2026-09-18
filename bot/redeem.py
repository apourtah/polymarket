"""Redeem resolved winning positions on Polygon for a Polymarket wallet.
Wallet kinds (sig_type as in py-clob-client): 0 = EOA holds the tokens; 1 = Polymarket proxy wallet (Magic/email login),
called through the ProxyWalletFactory; 2 = Gnosis Safe (browser-wallet login), called through execTransaction with the
owner's pre-validated signature. Gas is paid in POL by the EOA in every case. No Polymarket fee for redemption."""
import os, time, logging, requests
from web3 import Web3
from eth_abi import encode
log = logging.getLogger("evening")
RPC = os.environ.get("POLYGON_RPC", "https://polygon-rpc.com")
USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"; CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"; PROXY_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
DATA_API = "https://data-api.polymarket.com"
CTF_ABI = [{"name": "redeemPositions", "type": "function", "inputs": [{"name": "collateralToken", "type": "address"}, {"name": "parentCollectionId", "type": "bytes32"}, {"name": "conditionId", "type": "bytes32"}, {"name": "indexSets", "type": "uint256[]"}], "outputs": []},
           {"name": "isApprovedForAll", "type": "function", "stateMutability": "view", "inputs": [{"name": "owner", "type": "address"}, {"name": "operator", "type": "address"}], "outputs": [{"type": "bool"}]},
           {"name": "setApprovalForAll", "type": "function", "inputs": [{"name": "operator", "type": "address"}, {"name": "approved", "type": "bool"}], "outputs": []}]
NRA_ABI = [{"name": "redeemPositions", "type": "function", "inputs": [{"name": "_conditionId", "type": "bytes32"}, {"name": "_amounts", "type": "uint256[]"}], "outputs": []}]
FACTORY_ABI = [{"name": "proxy", "type": "function", "stateMutability": "payable", "inputs": [{"name": "calls", "type": "tuple[]", "components": [{"name": "typeCode", "type": "uint8"}, {"name": "to", "type": "address"}, {"name": "value", "type": "uint256"}, {"name": "data", "type": "bytes"}]}], "outputs": []}]
SAFE_ABI = [{"name": "execTransaction", "type": "function", "stateMutability": "payable", "inputs": [{"name": "to", "type": "address"}, {"name": "value", "type": "uint256"}, {"name": "data", "type": "bytes"}, {"name": "operation", "type": "uint8"}, {"name": "safeTxGas", "type": "uint256"}, {"name": "baseGas", "type": "uint256"}, {"name": "gasPrice", "type": "uint256"}, {"name": "gasToken", "type": "address"}, {"name": "refundReceiver", "type": "address"}, {"name": "signatures", "type": "bytes"}], "outputs": [{"type": "bool"}]}]

def redeemable_winners(address):
    """Resolved positions with a payout, from the data-api (redeemable=True also lists losers, so filter on value)."""
    r = requests.get(f"{DATA_API}/positions", params={"user": address, "redeemable": "true", "limit": 500}, timeout=30); r.raise_for_status()
    return [p for p in r.json() if p.get("redeemable") and float(p.get("currentValue") or 0) > 0.01 and float(p.get("size") or 0) > 0]

def _w3(): return Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 60}))

def _send(w3, acct, tx):
    tx.update({"from": acct.address, "nonce": w3.eth.get_transaction_count(acct.address), "chainId": 137})
    if "gas" not in tx: tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.3)
    fee = w3.eth.gas_price; tx.setdefault("maxFeePerGas", int(fee * 2)); tx.setdefault("maxPriorityFeePerGas", min(int(fee), Web3.to_wei(40, "gwei")))
    h = w3.eth.send_raw_transaction(acct.sign_transaction(tx).raw_transaction); rc = w3.eth.wait_for_transaction_receipt(h, timeout=240)
    if rc.status != 1: raise RuntimeError(f"tx {h.hex()} reverted")
    return h.hex()

def _route(w3, acct, wallet, inner_to, inner_data):
    """Wrap a call so it executes from the token holder: EOA directly, proxy via factory, Safe via execTransaction."""
    st = int(wallet.get("sig_type", 0)); holder = Web3.to_checksum_address(wallet["funder"] or acct.address)
    if st == 0: return {"to": inner_to, "data": inner_data, "value": 0}
    if st == 1:
        f = w3.eth.contract(address=Web3.to_checksum_address(PROXY_FACTORY), abi=FACTORY_ABI)
        return {"to": f.address, "data": f.encode_abi("proxy", args=[[(1, inner_to, 0, inner_data)]]), "value": 0}
    if st == 2:
        s = w3.eth.contract(address=holder, abi=SAFE_ABI); sig = encode(["address", "uint256"], [acct.address, 0]) + b"\x01"   # pre-validated: msg.sender is an owner
        return {"to": holder, "data": s.encode_abi("execTransaction", args=[inner_to, 0, inner_data, 0, 0, 0, 0, "0x" + "0" * 40, "0x" + "0" * 40, sig]), "value": 0}
    raise ValueError(f"unknown sig_type {st}")

def redeem_all(wallet, dry_run=True, max_positions=20):
    """Redeem every resolved winning position held by `wallet` ({key, funder, sig_type}). Returns (n, usd)."""
    from eth_account import Account
    acct = Account.from_key(wallet["key"]); holder = Web3.to_checksum_address(wallet.get("funder") or acct.address)
    wins = redeemable_winners(holder); usd = sum(float(p["currentValue"]) for p in wins)
    if not wins: return 0, 0.0
    log.info("redeem: %d resolved winner(s) worth $%.2f on %s%s", len(wins), usd, holder[:10], " [DRY]" if dry_run else "")
    if dry_run: return len(wins), usd
    w3 = _w3(); ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=CTF_ABI); nra = w3.eth.contract(address=Web3.to_checksum_address(NEG_RISK_ADAPTER), abi=NRA_ABI)
    if w3.eth.get_balance(acct.address) < Web3.to_wei(0.05, "ether"): log.error("redeem: EOA %s has < 0.05 POL for gas", acct.address); return 0, 0.0
    if any(p.get("negativeRisk") for p in wins) and not ctf.functions.isApprovedForAll(holder, nra.address).call():
        _send(w3, acct, _route(w3, acct, wallet, ctf.address, ctf.encode_abi("setApprovalForAll", args=[nra.address, True]))); log.info("redeem: approved NegRiskAdapter on CTF")
    done = 0; got = 0.0; seen = set()
    for p in wins[:max_positions]:
        cid = p["conditionId"]
        if cid in seen: continue
        seen.add(cid)
        try:
            if p.get("negativeRisk"):
                amt = int(round(float(p["size"]) * 1e6)); amounts = [amt, 0] if int(p.get("outcomeIndex", 0)) == 0 else [0, amt]
                data = nra.encode_abi("redeemPositions", args=[Web3.to_bytes(hexstr=cid), amounts]); to = nra.address
            else:
                data = ctf.encode_abi("redeemPositions", args=[Web3.to_checksum_address(USDC), b"\x00" * 32, Web3.to_bytes(hexstr=cid), [1, 2]]); to = ctf.address
            h = _send(w3, acct, _route(w3, acct, wallet, to, data)); done += 1; got += float(p["currentValue"])
            log.info("redeemed $%.2f  %s  tx %s", float(p["currentValue"]), p.get("title", cid)[:70], h[:12]); time.sleep(1)
        except Exception as ex: log.error("redeem failed for %s: %s", p.get("title", cid)[:60], ex)
    return done, got

def free_cash(client):
    """USDC balance (in $) of the client's funding wallet, per the CLOB balance endpoint."""
    from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
    try: return int(client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)).get("balance", 0)) / 1e6
    except Exception as ex: log.warning("balance check failed: %s", ex); return None
