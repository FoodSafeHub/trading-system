from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas.account import AccountSummary, Position, Quote
from app.services.brokers.factory import get_broker

router = APIRouter(prefix="/account", tags=["account"])


@router.get("/summary", response_model=List[AccountSummary])
async def get_account_summary():
    broker = get_broker()
    await broker.authenticate()
    return await broker.get_accounts()


@router.get("/positions", response_model=List[Position])
async def get_positions(account_id: str = ""):
    broker = get_broker()
    await broker.authenticate()
    accounts = await broker.get_accounts()
    if not accounts:
        raise HTTPException(status_code=404, detail="No accounts found")
    acct_id = account_id or accounts[0].account_id
    return await broker.get_positions(acct_id)


@router.get("/quotes", response_model=dict)
async def get_quotes(symbols: str):
    """Pass comma-separated symbols: ?symbols=AAPL,MSFT"""
    broker = get_broker()
    await broker.authenticate()
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return await broker.get_quotes(sym_list)
