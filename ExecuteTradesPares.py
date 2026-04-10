from TradesPares import (
    fetch_pair,
    calc_hedge_ratio,
    calc_spread,
    calc_zscore,
    test_cointegration,
    get_signal,
    get_position_qty,
    run_backtest,
    execute_pair_order,
)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

# CONFIGURACIÓN
API_KEY    = "PKIQLVPLYKNHZ62EXVZDP2EHPR"
API_SECRET = "2hqNbznjYjV7QZpPjmvyPrEgwXCWfKuCg8RhbasiKkLF"

SYMBOL_A = "SPY"
SYMBOL_B = "QQQ"

ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD  = 0.5
ZSCORE_WINDOW   = 30
YEARS_DATA      = 2
INITIAL_CAPITAL = 100_000

# CLIENTES ALPACA
data_client    = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)


def main():
    # 1. Descargar datos
    print("[ 1 ] Descargando datos...")
    df = fetch_pair(SYMBOL_A, SYMBOL_B, data_client, years=YEARS_DATA)

    # 2. Hedge ratio
    print("[ 2 ] Calculando hedge ratio...")
    hedge_ratio = calc_hedge_ratio(df[SYMBOL_A], df[SYMBOL_B])
    print(f"  Hedge Ratio: {hedge_ratio:.4f}")
    print(f"  Por cada 1 accion de {SYMBOL_A}, necesitas {hedge_ratio:.2f} de {SYMBOL_B}")

    # 3. Spread y z-score
    print("[ 3 ] Construyendo spread y z-score...")
    spread = calc_spread(df[SYMBOL_A], df[SYMBOL_B], hedge_ratio)
    zscore = calc_zscore(spread, window=ZSCORE_WINDOW)
    print(f"  Media del spread:   ${spread.mean():.2f}")
    print(f"  Std del spread:     ${spread.std():.2f}")
    print(f"  Z-score actual:     {float(zscore.iloc[-1]):+.3f}")

    # 4. Test de cointegracion
    print("[ 4 ] Test de cointegracion...")
    coint = test_cointegration(spread, SYMBOL_A, SYMBOL_B)

    if not coint["is_cointegrated"]:
        print(f"  ADVERTENCIA: El par no esta cointegrado (p={coint['p_value']:.4f})")
        print(f"  Considera probar otro par antes de operar.")

    # 5. Backtester walk-forward
    print("[ 5 ] Corriendo backtester walk-forward...")
    results_df, trades_df, _ = run_backtest(
        df,
        SYMBOL_A,
        SYMBOL_B,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        zscore_window=ZSCORE_WINDOW,
        initial_capital=INITIAL_CAPITAL,
    )

    # Filtro de calidad — no operar si el backtest no pasa los mínimos
    BACKTEST_MIN_SHARPE = 0.5
    BACKTEST_MIN_RETURN = 0.0

    final_value    = results_df["portfolio_value"].iloc[-1]
    total_return   = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
    daily_rets     = results_df["portfolio_value"].pct_change()
    sharpe         = (
        daily_rets.mean() / daily_rets.std() * (252 ** 0.5)
        if daily_rets.std() > 0 else 0
    )

    backtest_ok = sharpe >= BACKTEST_MIN_SHARPE and total_return >= BACKTEST_MIN_RETURN

    if not backtest_ok:
        print(f"\n  ⛔ Backtest no pasa el filtro minimo")
        print(f"     Sharpe:  {sharpe:.3f}  (minimo {BACKTEST_MIN_SHARPE})")
        print(f"     Retorno: {total_return*100:.2f}%  (minimo 0%)")
        print(f"     Sin orden ejecutada — revisa los parametros o el par.")
        return

    # 6. Señal del dia
    zscore_now  = float(zscore.iloc[-1])
    price_a_now = float(df[SYMBOL_A].iloc[-1])
    price_b_now = float(df[SYMBOL_B].iloc[-1])

    print(f"  Z-score actual:   {zscore_now:>+.3f}")
    print(f"  {SYMBOL_A} precio:      ${price_a_now:.2f}")
    print(f"  {SYMBOL_B} precio:       ${price_b_now:.2f}")
    print(f"  Spread actual:    ${float(spread.iloc[-1]):.2f}")

    qty_a = get_position_qty(SYMBOL_A, trading_client)
    qty_b = get_position_qty(SYMBOL_B, trading_client)

    if qty_a > 0 and qty_b < 0:
        current_position = "LONG_A_SHORT_B"
    elif qty_a < 0 and qty_b > 0:
        current_position = "SHORT_A_LONG_B"
    else:
        current_position = "FLAT"

    print(f"  Posicion actual:  {current_position}")

    signal = get_signal(zscore_now, ENTRY_THRESHOLD, EXIT_THRESHOLD, current_position)
    print(f"  Señal:  {signal}")

    if signal == "LONG_A_SHORT_B":
        print(f"  COMPRAR {SYMBOL_A}  +  VENDER (short) {SYMBOL_B}")
    elif signal == "SHORT_A_LONG_B":
        print(f"  VENDER (short) {SYMBOL_A}  +  COMPRAR {SYMBOL_B}")
    elif signal == "CLOSE":
        print(f"  Cerrar posicion actual — spread regreso a la media")
    else:
        print(f"  Z-score en zona neutral ({zscore_now:+.2f}) — sin accion")

    # 7. Ejecutar orden
    if signal != "HOLD":
        print("[ 7 ] Ejecutando orden...")
        execute_pair_order(
            signal, SYMBOL_A, SYMBOL_B,
            price_a_now, price_b_now,
            hedge_ratio, trading_client,
        )
    else:
        print("[ 7 ] Sin orden — HOLD")


if __name__ == "__main__":
    main()
