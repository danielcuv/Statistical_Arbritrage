# Statistical Arbitrage — Pairs Trading (SPY / QQQ)

Sistema de trading algorítmico basado en **Mean Reversion** con pares de activos cointegrados.
Construido desde cero en Python como parte de un journey de aprendizaje en Quantitative Finance.

---

## ¿Qué hace?

Detecta divergencias estadísticas entre dos ETFs correlacionados (SPY y QQQ) y opera automáticamente
cuando el spread entre ambos se aleja demasiado de su media histórica — apostando a que convergerá.

```
SPY sube más que QQQ  →  SHORT SPY + LONG QQQ
QQQ sube más que SPY  →  LONG SPY  + SHORT QQQ
Spread regresa a la media  →  Cerrar posición y tomar ganancia
```

---

## Resultados (backtest out-of-sample, sept 2025 → abr 2026)

| Métrica | Estrategia | Buy & Hold SPY |
|---|---|---|
| Retorno total | +3.17% | +5.63% |
| Retorno anualizado | +5.34% | — |
| **Sharpe Ratio** | **2.648** | ~0.8–1.0 |
| **Max Drawdown** | **-0.46%** | ~-15% a -20% |
| Win Rate | 100% (4 trades) | — |

> La estrategia no compite en retorno bruto con el benchmark — es **market neutral**.
> Su ventaja está en el perfil de riesgo: Sharpe 2.648 con un drawdown de apenas -0.46%.

---

## Pipeline

```
[ 1 ] fetch_pair()          →  Descarga y alinea precios de cierre de SPY y QQQ
[ 2 ] calc_hedge_ratio()    →  OLS regression para encontrar la proporción estacionaria
[ 3 ] calc_spread/zscore()  →  Spread y z-score con ventana rodante de 30 días
[ 4 ] test_cointegration()  →  ADF test (p-value < 0.05 = fundamento estadístico válido)
[ 5 ] run_backtest()        →  Walk-forward 70/30 con filtro de calidad pre-ejecución
[ 6 ] get_signal()          →  LONG_A_SHORT_B / SHORT_A_LONG_B / CLOSE / HOLD
[ 7 ] execute_pair_order()  →  Ambas patas simultáneas vía Alpaca API (paper trading)
```

---

## Conceptos clave implementados

**Cointegración:** dos activos cuya *diferencia* es estacionaria — más robusto que correlación.

**Hedge Ratio (OLS):**
```
precio_SPY = β × precio_QQQ + α + ε
β = 0.9123  →  por cada 1 acción de SPY, se necesitan 0.91 de QQQ
```

**Z-Score del spread:**
```
z = (spread_hoy - media_rolling_30d) / std_rolling_30d

z > +2.0  →  SHORT SPY  + LONG QQQ
z < -2.0  →  LONG  SPY  + SHORT QQQ
|z| < 0.5 →  Cerrar posición
```

**Filtro de calidad:** antes de ejecutar cualquier orden, el sistema verifica que el backtest
supere un Sharpe mínimo de 0.5 y retorno positivo. Si no, no opera.

---

## Estructura del proyecto

```
Statistical_Arbitrage/
├── TradesPares.py          # Motor: cálculos, backtester, ejecución de órdenes
├── ExecuteTradesPares.py   # Punto de entrada, configuración, flujo principal
├── README.md
└── LICENSE
```

---

## Instalación

```bash
# Clonar el repo
git clone https://github.com/danielcuv/Statistical_Arbitrage.git
cd Statistical_Arbitrage

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install alpaca-py statsmodels pandas numpy
```

---

## Configuración

En `ExecuteTradesPares.py`, edita las siguientes variables:

```python
API_KEY    = "tu_api_key_de_alpaca"
API_SECRET = "tu_api_secret_de_alpaca"

SYMBOL_A = "SPY"     # activo principal
SYMBOL_B = "QQQ"     # activo de cobertura

ENTRY_THRESHOLD = 2.0   # z-score para abrir posición
EXIT_THRESHOLD  = 0.5   # z-score para cerrar posición
ZSCORE_WINDOW   = 30    # ventana rodante en días
YEARS_DATA      = 2     # años de historial a descargar
```

> ⚠️ Nunca subas tus API keys a GitHub. Usa variables de entorno o un archivo `.env`.

---

## Uso

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar
python ExecuteTradesPares.py
```

El script descarga datos, valida el par estadísticamente, corre el backtester,
y si pasa el filtro de calidad, genera la señal y ejecuta la orden en paper trading.

---

## Pares probados

| Par | ADF p-value | Sharpe | Max DD | Resultado |
|---|---|---|---|---|
| NVDA / AMD | 0.0138 | 1.132 | -5.74% | Descartado (split 10:1 de NVDA en jun 2024 distorsiona el hedge ratio) |
| **SPY / QQQ** | **0.0409** | **2.648** | **-0.46%** | **✅ Par activo** |

---

## Stack

- **Python 3.9**
- **pandas / NumPy** — manipulación de datos
- **statsmodels** — OLS regression, ADF test
- **alpaca-py** — datos históricos y ejecución de órdenes (paper trading)

---

## Disclaimer

Este proyecto es **exclusivamente educativo**. No constituye asesoramiento financiero.
Todas las operaciones se ejecutan en **paper trading** (dinero simulado).
El rendimiento pasado no garantiza resultados futuros.

---

## Autor

**Daniel** — aprendiendo Quantitative Finance construyendo cada estrategia desde los principios.

Proyecto parte de una serie progresiva:
1. ML Predictor con Random Forest → [ML_PredictMarket](https://github.com/danielcuv/ML_PredictMarket)
2. **Statistical Arbitrage (este repo)**

