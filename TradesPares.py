# Statistical Arbitrage
# Motor de back engine integrado.

# DEscarga datos de dos activos correlacionados(en este caso sera NVvidia y AMD)
# Calcula el Hedge ratio con OLS para contruir un spread estacionario
# (coeficiente estacionario) propeorcion de una inversion que esta protegida contra fluctuaiones adversas del mercado.

# Verifica cointegraciones con el test ADF (p-value < 0.05)
# Dickey-Fuller-Aumentada test estadistico utiliado para definir si una serie es estacionaria en un analisi de series temporales

# Normalizar el spread con z-score

# Genera la señales: z > + 2 SHORT A / LONG B | z < - 2 -> LONG A / SHORT B

# Corre un backtest walk-foward para validar historicamente
# Y ejecuta las ordenes en ALpaca paper

# pip install alpaca-py scikit-learn statsmodels.

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# DESCARGAR LOS DATOS

def fetch_data(symbol:str, data_client: StockHistoricalDataClient, years: int = 2) -> pd.DataFrame:
    # Descargar las barras en un OHLCH que esto es un estandar financiero fundamental
    # Open, High, Low, Close, Volume

    print(f"    descargar los simbolos de {symbol}...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days = 365 * years)

    request = StockBarsRequest(
        symbol_or_symbols = symbol,
        timeframe = TimeFrame.Day,
        start = start, end = end,
        feed = "iex",
    )

    bars = data_client.get_stock_bars(request)
    df = bars.df.reset_index()

    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()
    if "timestamp" in df.columns:
        df = df.rename(columns = {"timestamp": "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop = True)
    return df[["date", "close"]].rename(columns = {"close": symbol})

def fetch_pair(symbol_a: str, symbol_b: str, data_client: StockHistoricalDataClient, years: int = 2) -> pd.DataFrame:
    # Se descargan los precios de cierre de los simbolos.
    # Solo conserva fechas donde ambos tienen datos (inner join)
    df_a = fetch_data(symbol_a, data_client, years)
    df_b = fetch_data(symbol_b, data_client, years)

    df = pd.merge(df_a, df_b, on = "date", how = "inner")
    df = df.dropna().reset_index(drop = True)
    print(f"    Par alineado: {len(df)} dias en comun")
    return df

# Hedge ratio con OLS

def calc_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    # Calcular el Hedge ratio B por meido de una OLS:
    # El hedge ratio nos dice cunatas unidades de B necesitamos para neutralizar el movimiento de 1 unidad de A.
    #Ejemplito: B = 0.85 significa que por cada accion de NVDA, se necesitan de 0.85 acciones de AMD para que el spread (modelo) sea estacionario
    X = add_constant(price_b)
    model = OLS(price_a, X).fit()
    beta = model.params.iloc[1]     # Esto es el coeficiente de price_b
    return float(beta)

#Spread y Z-SCORE

def calc_spread(price_a: pd.Series, price_b: pd.Series, hedge_ratio: float) -> pd.Series:
    # formula para sacar el spread:
    # SPREAD = precio_A - Hedge_ratio * price_B
    # este spread deberia ser estacionnario si los activos estan cointegrados, que estos oscilen alrededor de una medida sin alejarse indefinidamente
    return price_a - hedge_ratio * price_b

def calc_zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    # Z-Score del spread con ventana rodante:
    # z = (spread_hoy - media_rolling) / std_rolling
    # La ventana rodante permite que z-score se adapte gradualmente a cambios de regimen de mercado en lugar de usar una media fija
    # Como se imterpreta:
    # z > +2 -> spread demasiado alto(A caro vs B) -> Short A, Long B
    # z < -2 -> spread demasiado bajo (A barato vs B) -> Long A, Short B
    # |z| < 0.5 -> spread en zona neutral -> cerrar posicion
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore

# Test de Cointegracion (ADF)

def test_cointegration(spread: pd.Series, symbol_a: str, symbol_b: str) -> dict:
    # Augmented Dickey-Fuller           Que es?
    # H0: spred tiene raiz unitaria -> No es estacionario -> sin mean de reversion
    # H1: el spread es estacionario -> Si hay reversion -> podemos operar
    # Se rechaa H0 si p-value < 0.05 (95% de confianza).
    # El test tambien reporta los valores criticos al 1%, 5% y 10%,
    # Y el ADF statistic - cuanto mas negativo, mas evidencia de estacionariedad
    result = adfuller(spread.dropna(), autolag = "AIC")

    adf_stat = result[0]
    p_value = result[1]
    crit_vals = result[4]
    is_cointegrated = p_value < 0.05

    print(f"    test de cointegracion - {symbol_a} / {symbol_b}")
    print(f"    ADF statistic: {adf_stat:.4f}")
    print(f"    P-Value: {p_value:.4f}")
    print(f"    critico 1%: {crit_vals['1%']:.4f}")
    print(f"    critico 5%: {crit_vals['5%']:.4f}")
    print(f"    critico 10%: {crit_vals['10%']:.4f}")

    if is_cointegrated:
        print(f"    Cointegrados - p = {p_value:.4f} < 0.05 -> spread estacionario")
        print(f"    La estrategia tiene fundamento estadistico.")
    else:
        print(f"    No cointegrados - p = {p_value:.4f} >= 0.05 -> spread no estacionario")
        print(f"    waters(aguas) - Sin garantia de reversion")
    
    return{
        "adf_stat": adf_stat,
        "p_value": p_value,
        "is_cointegrated": is_cointegrated,
        "crit_vals": crit_vals,
    }

# Señal para tradear

def get_signal(zscore_now: float, entry_threshold: float, exit_threshold: float, current_position: str) -> str:
    # Genera la señal del dia basandose en el z-score actual
    # Posibles posiciones:
    # 'Long_A_Short_B' -> se compra A, se vende B (spread bajo)
    # 'Short_A_Long_B' -> se vende A, se compra B (spread alto)
    # 'Flat' -> sin posicion
    # La logica de entrada/salida:          Teniendo una posicion:
    # z > + entry -> abrir Short_A_Long_B
    # z < - entry -> abrir Long_A_Short_B
    # Con la ejecucion de la posicion
    # |z| < exit -> cerrar (spread regreso a la media -> tomar ganancia)

    if current_position == "FLAT":
        if zscore_now > entry_threshold:
            return "SHORT_A_LONG_B"
        elif zscore_now < - entry_threshold:
            return "LONG_A_SHORT_B"
        else:
            return "HOLD"

    elif current_position == "LONG_A_SHORT_B":
        if abs(zscore_now) < exit_threshold:
            return "CLOSE"
        else:
            return "HOLD"

    elif current_position == "SHORT_A_LONG_B":
        if abs(zscore_now) < exit_threshold:
            return "CLOSE"
        else:
            return "HOLD"

    return "HOLD"

#Backtester del WALK-FOWARD

def run_backtest(df: pd.DataFrame,
                 symbol_a: str, symbol_b: str,
                 entry_threshold: float = 2.0,
                 exit_threshold:  float = 0.5,
                 zscore_window:   int   = 30,
                 train_ratio:     float = 0.70,
                 initial_capital: float = 100_000) -> pd.DataFrame:
    # Este backtest sirve para esta estrategia de pares
    # Divide los datos en 2:
        # train (70%): calcula hadge ratio y parametros del spread
        # test (30%): simula operaciones dia a dia (out-of-sample)
    #Logica del capital disponible
        # En cada Trade, asigna el 50% del capital a cada pata
        # Long A + Short B: compra A con 50% del capital, vende B con 50%
        # Short A + Long B: vende A con 50% del capital, compra B con 50%
        # El PnL se calcula cuando se cierra la posicion
    # Metricas reportadas: retorno total, anualziado, Sharpe, Max Drawdown,
    # Win Rate, numero de operaciones completas, vs Buy & Hold de A
    print(f"    backtest - {symbol_a} / {symbol_b}")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop = True)
    test_df = df.iloc[split_idx:].reset_index(drop = True)

    #Hedge ratio calculado SOLO con datos de entrenamiento
    hedge_ratio = calc_hedge_ratio(train_df[symbol_a], train_df[symbol_b])
    print(f"    Hedge Ratio (OLS train): {hedge_ratio:.4f}")
    print(f"    Periodo test: {test_df['date'].iloc[0].date()} -> {test_df['date'].iloc[-1].date()}")

    #Construir spread y z-score en el preiodo de test
    spread = calc_spread(test_df[symbol_a], test_df[symbol_b], hedge_ratio)
    zscore = calc_zscore(spread, window = zscore_window)

    #Simulacion del dia a dia
    capital = initial_capital
    position = "FLAT"
    entry_price_a = entry_price_b = 0.0
    entry_shares_a = entry_shares_b = 0.0
    entry_side_a = "" # "Long" o "Short"

    portfolio_values = []
    trades = []

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        price_a = row[symbol_a]
        price_b = row[symbol_b]
        z = zscore.iloc[i]

        if pd.isna(z):
            portfolio_values.append({"date": row["date"], "portfolio_value": capital})
            continue

        signal = get_signal(z, entry_threshold, exit_threshold, position)
        # Ejecuta la operacion
        if signal in ("LONG_A_SHORT_B", "SHORT_A_LONG_B") and position == "FLAT":
            alloc = capital * 0.50      # que esto es de 50% por pata

            entry_shares_a = alloc / price_a
            entry_shares_b = alloc / price_b
            entry_price_a = price_a
            entry_price_b = price_b
            entry_side_a = "LONG" if signal == "LONG_A_SHORT_B" else "SHORT"
            position = signal

            trades.append({
                "date": row["date"], "action": "OPEN",
                "side_a": entry_side_a, "price_a": price_a, "price_b": price_b,
                "zscore": z,
            })

        elif signal == "CLOSE" and position != "FLAT":
            # PnL de la pata A
            if entry_side_a == "LONG":
                pnl_a = (price_a - entry_price_a) * entry_shares_a
                pnl_b = (entry_price_b - price_b) * entry_shares_b
            else:
                pnl_a = (entry_price_a - price_a) * entry_shares_a
                pnl_b = (price_b - entry_price_b) * entry_shares_b

            total_pnl = pnl_a + pnl_b
            capital += total_pnl

            trades.append({
                "date": row["date"], "action": "CLOSE",
                "side_a": entry_side_a, "price_a": price_a, "price_b": price_b,
                "pnl": total_pnl, "zscore": z,
            })

            position = "FLAT"

        # Valor del portafolio (mark-tomarket si hay posicion abierta)
        if position != "FLAT":
            if entry_side_a == "LONG":
                mtm = ((price_a - entry_price_a) * entry_shares_a + (entry_price_b - price_b) * entry_shares_b)
            else:
                mtm = ((entry_price_a - price_a) * entry_shares_a + (price_b - entry_price_b) * entry_shares_b)
            portfolio_value = capital + mtm
        else:
            portfolio_value = capital

        portfolio_values.append({"date": row["date"], "portfolio_value": portfolio_value})

    result_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades)

    # Metricas
    final_value = result_df["portfolio_value"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    n_days = len(result_df)
    ann_ret = (1 + total_return) ** (252 / n_days) - 1

    result_df["daily_return"] = result_df["portfolio_value"].pct_change()
    sharpe = (
        result_df["daily_return"].mean() /
        result_df["daily_return"].std() * np.sqrt(252)
        if result_df["daily_return"].std() > 0 else 0
    )

    roll_max = result_df["portfolio_value"].cummax()
    max_dd = ((result_df["portfolio_value"] - roll_max) / roll_max).min()

    closed_trades = trades_df[trades_df["action"] == "CLOSE"]if not trades_df.empty else pd.DataFrame()
    n_trades = len(closed_trades)
    win_rate = 0.0
    if n_trades > 0:
        win_rate = (closed_trades["pnl"] > 0).mean()

    bh_return = (test_df[symbol_a].iloc[-1] - test_df[symbol_a].iloc[0]) / test_df[symbol_a].iloc[0]

    #imprime el resumen

    print(f"    Capital inicial: ${initial_capital:>12,.2f}")
    print(f"    Capital final: ${final_value:>12,.2f}")
    print(f"    Retorno total: {total_return * 100:>+.2f}%")
    print(f"    Retorno anualizado: {ann_ret * 100:>+.2f}%")
    print(f"    Sharpe Ratio: {sharpe:>8.3f}")
    print(f"    Max Drawdown: {max_dd * 100:> 8.2f}%")
    print(f"    Trades completos: {n_trades}")
    print(f"    Win rate: {win_rate * 100:>8.1f}%")
    print(f"    Buy & hold({symbol_a}): {bh_return * 100:>+.2f}% (benchmark)")

    if total_return > bh_return:
        print(f"    supera el Buy & Hold por unn ttal de {(total_return - bh_return) * 100:.2f}%")
    else:
        print(f"    Por debajo del Buy & Hold en {(bh_return - total_return) * 100:.2f}%")

    return result_df, trades_df, hedge_ratio

# Ejecucion de ordenes (ambas patas)

def get_position_qty(symbol: str, trading_client: TradingClient) -> float:
    try:
        pos = trading_client.get_open_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0

def execute_pair_order(signal: str, symbol_a: str, symbol_b: str, price_a: float, price_b: float, hedge_ratio: float, trading_client: TradingClient):
    # Ejecuta las patas del trade simultaneamente.
    # LONG_A_SHORT_B
        # -Compra symbol_a con 50% del cash disponible
        # -Vende symbol_b con 50% del cash disponible (short)
    # SHORT_A_LONG_B
        # -Vende symbol_a (short) con 50% del cash
        # -Compra symbol_b con 50% del cash
    # CLOSE
        # -Cierra todas las posiciones abiertas en ambos símbolos

    account = trading_client.get_account()
    cash = float(account.cash)
    alloc = cash * 0.50

    def place_order(symbol, qty, side):
        if qty < 1:
            print(f"    {symbol} qty < 1, sin orden")
            return
        order = MarketOrderRequest(
            symbol = symbol,
            qty = int(qty),
            side = side,
            time_in_force = TimeInForce.DAY
        )
        result = trading_client.submit_order(order)
        side_str  = "COMPRA" if side == OrderSide.BUY else "VENTA"
        price_map = {symbol_a: price_a, symbol_b: price_b}
        notional  = int(qty) * price_map.get(symbol, 0)
        print(f"    Orden {side_str} ejecutada | {int(qty)} * {symbol} (~${notional:,.2f})")
        print(f"    ID: {result.id}")

    if signal == "LONG_A_SHORT_B":
        print(f"\n Abriendo LONG {symbol_a} / SHORT {symbol_b}")
        qty_a = alloc / price_a
        qty_b = alloc / price_b
        place_order(symbol_a, qty_a, OrderSide.BUY)
        place_order(symbol_b, qty_b, OrderSide.SELL)

    elif signal == "SHORT_A_LONG_B":
        print(f"\n  Abriendo SHORT {symbol_a} / {symbol_b}")
        qty_a = alloc / price_a
        qty_b = alloc / price_b
        place_order(symbol_a, qty_a, OrderSide.SELL)
        place_order(symbol_b, qty_b, OrderSide.BUY)

    elif signal == "CLOSE":
        print(f"\n  Cerrando posicion del par")
        pos_a = get_position_qty(symbol_a, trading_client)
        pos_b = get_position_qty(symbol_b, trading_client)

        if pos_a > 0:
            place_order(symbol_a, abs(pos_a), OrderSide.SELL)
        elif pos_a < 0:
            place_order(symbol_a, abs(pos_a), OrderSide.BUY)

        if pos_b > 0:
            place_order(symbol_b, abs(pos_b), OrderSide.SELL)
        elif pos_b < 0:
            place_order(symbol_b, abs(pos_b), OrderSide.BUY)

    else:
        print(f"    Señal HOLD - sin orden")
        