# --- TOKEN UNIVERSE ---
MASTER_TOKEN_LIST = {
    "BTC": {"santiment_slug": "bitcoin", "coinapi_symbol_id": "BINANCE_SPOT_BTC_USDT"},
    "ETH": {"santiment_slug": "ethereum", "coinapi_symbol_id": "BINANCE_SPOT_ETH_USDT"},
    "MATIC": {
        "santiment_slug": "matic-network",
        "coinapi_symbol_id": "BINANCE_SPOT_MATIC_USDT",
    },
    "AAVE": {"santiment_slug": "aave", "coinapi_symbol_id": "BINANCE_SPOT_AAVE_USDT"},
    "SHIB": {
        "santiment_slug": "shiba-inu",
        "coinapi_symbol_id": "BINANCE_SPOT_SHIB_USDT",
    },
    # 'UNI': {'santiment_slug': 'uniswap', 'coinapi_symbol_id': 'BINANCE_SPOT_UNI_USDT'},
    # 'MKR': {'santiment_slug': 'maker', 'coinapi_symbol_id': 'BINANCE_SPOT_MKR_USDT'},
    # 'LDO': {'santiment_slug': 'lido-dao', 'coinapi_symbol_id': 'BINANCE_SPOT_LDO_USDT'},
    # 'IMX': {'santiment_slug': 'immutable-x', 'coinapi_symbol_id': 'BINANCE_SPOT_IMX_USDT'},
    # 'MANA': {'santiment_slug': 'decentraland', 'coinapi_symbol_id': 'BINANCE_SPOT_MANA_USDT'},
    # 'SNX': {'santiment_slug': 'synthetix-network-token', 'coinapi_symbol_id': 'BINANCE_SPOT_SNX_USDT'},
    # 'SAND': {'santiment_slug': 'the-sandbox', 'coinapi_symbol_id': 'BINANCE_SPOT_SAND_USDT'},
    # 'AXS': {'santiment_slug': 'axie-infinity', 'coinapi_symbol_id': 'BINANCE_SPOT_AXS_USDT'},
    # 'BAT': {'santiment_slug': 'basic-attention-token', 'coinapi_symbol_id': 'BINANCE_SPOT_BAT_USDT'},
    # 'QNT': {'santiment_slug': 'quant', 'coinapi_symbol_id': 'BINANCE_SPOT_QNT_USDT'},
    # 'COMP': {'santiment_slug': 'compound', 'coinapi_symbol_id': 'BINANCE_SPOT_COMP_USDT'},
    # 'CHZ': {'santiment_slug': 'chiliz', 'coinapi_symbol_id': 'BINANCE_SPOT_CHZ_USDT'},
    # 'RAY': {'santiment_slug': 'raydium', 'coinapi_symbol_id': 'BINANCE_SPOT_RAY_USDT'},
    # 'PAXG': {'santiment_slug': 'pax-gold', 'coinapi_symbol_id': 'BINANCE_SPOT_PAXG_USDT'},
    # 'RNDR': {'santiment_slug': 'render', 'coinapi_symbol_id': 'BINANCE_SPOT_RNDR_USDT'},
    # 'PEPE': {'santiment_slug': 'pepe', 'coinapi_symbol_id': 'BINANCE_SPOT_PEPE_USDT'},
    # 'INJ': {'santiment_slug': 'injective-protocol', 'coinapi_symbol_id': 'BINANCE_SPOT_INJ_USDT'},
    # 'ENS': {'santiment_slug': 'ethereum-name-service', 'coinapi_symbol_id': 'BINANCE_SPOT_ENS_USDT'},
    # 'GRT': {'santiment_slug': 'the-graph', 'coinapi_symbol_id': 'BINANCE_SPOT_GRT_USDT'},
}

# --- FEATURE PARAMETERS (HORIZON-SPECIFIC) ---
# These will be dynamically populated later and updated by the operator.
FEATURE_PARAMS = {
    "1h": {
        "rsi_15m": {"window": 14},
        "atr_15m": {"window": 40},
        "atr_1h": {"window": 40},
        "atr_4h": {"window": 14},
        "atr_1d": {"window": 40},
        "adx_15m": {"window": 10},
        "adx_1h": {"window": 10},
        "adx_4h": {"window": 10},
        "adx_1d": {"window": 10},
        "atr_normalized_15m": {},  # No parameters needed
    },
    "4h": {
        "rsi_15m": {"window": 14},
        "atr_15m": {"window": 40},
        "atr_1h": {"window": 40},
        "atr_4h": {"window": 14},
        "atr_1d": {"window": 40},
        "adx_15m": {"window": 20},
        "adx_1h": {"window": 20},
        "adx_4h": {"window": 20},
        "adx_1d": {"window": 20},
        "atr_normalized_15m": {},
    },
    "8h": {
        "rsi_15m": {"window": 14},
        "atr_15m": {"window": 40},
        "atr_1h": {"window": 40},
        "atr_4h": {"window": 14},
        "atr_1d": {"window": 40},
        "adx_15m": {"window": 30},
        "adx_1h": {"window": 30},
        "adx_4h": {"window": 40},
        "adx_1d": {"window": 30},
        "atr_normalized_15m": {},
    },
}

# --- TELEGRAM NOTIFICATION SETTINGS ---
TELEGRAM_ENABLED = True  # Master switch for all notifications

NOTIFICATION_SETTINGS = {
    "ON_POSITION_OPEN": True,
    "ON_POSITION_CLOSE": True,
    "DAILY_PNL_SUMMARY": True,
    "WEEKLY_PNL_SUMMARY": True,
    "SYSTEM_ALERTS": True,
}

# --- MLOPS & DRIFT DETECTION ---
DRIFT_ACTION_THRESHOLDS = {
    "MODERATE_DRIFT_SCORE": 0.3,  # Triggers Retraining
    "SEVERE_DRIFT_SCORE": 0.6,  # Triggers Kill-Switch
}

# --- LOGGING ---
LOG_LEVEL = "DEBUG"
LOG_FILE_PATH = "logs/unicorn_wealth.log"

# --- DATABASE ---
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
DB_POOL_RECYCLE_SECONDS = 3600

# --- EXCHANGE CONNECTION SETTINGS ---
EXCHANGE_CONNECTION_SETTINGS = {
    "binance": {"unicorn": True, "personal": False},
    "kraken": {"unicorn": False, "personal": False},
    "hyperliquid": {"unicorn": False, "personal": False},
    "bitget": {"unicorn": False, "personal": False},
    "bybit": {"unicorn": False, "personal": False},
    "kucoin": {"unicorn": False, "personal": False},
}

# --- API CLIENT SETTINGS ---
# Unified client settings for API consumers (e.g., Santiment client)
API_CLIENT_SETTINGS = {
    # Tenacity configuration used by async API clients (flat keys for new clients)
    "TENACITY_MAX_ATTEMPTS": 3,
    "TENACITY_WAIT_MULTIPLIER": 1,
    # Simple circuit breaker configuration (flat keys for new clients)
    "CIRCUIT_BREAKER_MAX_FAILURES": 5,
    "CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS": 60,
    # Backward-compatible nested structure for existing clients/tests
    "TENACITY_RETRY": {
        "WAIT_MIN": 1,
        "WAIT_MAX": 60,
        "STOP_MAX_ATTEMPT": 5,
    },
    "CIRCUIT_BREAKER": {
        "FAIL_MAX": 5,
        "RESET_TIMEOUT": 60,
    },
}

# --- ML TRAINING SETTINGS ---
# Split proportions are relative to the total dataset size.
# Train portion is TRAIN_TEST_SPLIT_PCT minus VALIDATION_SPLIT_PCT, validation is VALIDATION_SPLIT_PCT,
# and the remainder is the test set.
TRAIN_TEST_SPLIT_PCT = 0.8
VALIDATION_SPLIT_PCT = 0.1  # of the total dataset, not the training set
TIME_DECAY_HALF_LIFE_MONTHS = 6
TIME_DECAY_WEIGHT_FLOOR = 0.1

# --- BACKTESTING SETTINGS ---
BACKTESTING_SETTINGS = {
    "SLIPPAGE_PCT": 0.05,  # per fill, percent
    "TAKER_FEE": 0.04,  # per fill, percent
    # Challenger Sharpe must be 10% higher than Champion to be promoted
    "CHAMPION_PROMOTION_THRESHOLD_PCT": 10.0,
}

# --- DATA LIFECYCLE SETTINGS ---
# The default number of days of historical data to download for each token.
HISTORICAL_LOOKBACK_DAYS = 730  # (Approx. 2 years)

# --- WALK-FORWARD OPTIMIZATION SETTINGS (in days) ---
# The size of the initial training window.
WFO_TRAINING_WINDOW_DAYS = 365

# The size of the out-of-sample validation window.
WFO_VALIDATION_WINDOW_DAYS = 30

# The number of days to slide the window forward for the next fold.
WFO_STEP_SIZE_DAYS = 30

# --- EXECUTION STRATEGY SETTINGS ---
ATR_STOP_LOSS_MULTIPLIER = 2.5
USE_FIXED_TAKE_PROFIT = False  # If False, exit is based on opposing signal.
FIXED_TAKE_PROFIT_PCT = (
    3.0  # Target profit percentage if USE_FIXED_TAKE_PROFIT is True.
)

# Position sizing and leverage
POSITION_SIZING_MODE = (
    "percent_risk"  # Options: 'percent_risk', 'percent_fixed', 'usd_fixed'
)
RISK_PERCENT = 1.0  # Used for 'percent_risk' mode (percent of balance at risk)
FIXED_POSITION_PERCENT = 5.0  # Used for 'percent_fixed' mode (percent of balance)
FIXED_POSITION_USD = 1000.0  # Used for 'usd_fixed' mode
MAX_POSITION_SIZE_USD = 20000.0  # Hard cap on notional size per position
LEVERAGE = 5  # Desired leverage on the exchange

# Blueprint-derived thresholds and weights
MINIMUM_MOVE_PCT = (
    0.5  # Minimum expected move (percent) for a model prediction to be viable
)
MIN_CONSENSUS_PROBABILITY_THRESHOLD = 0.55  # Minimum consensus probability to act
MODEL_WEIGHTS = {
    # Example model weights; override via operator settings as needed
    "model_a": 1.0,
    "model_b": 1.0,
}

# Mapping of outcome labels to expected percentage moves used by viability filter
CLASS_TO_EXPECTED_MOVE_PCT = {
    "OPEN_LONG": 1.0,
    "OPEN_SHORT": 1.0,
    "EXIT_LONG": 0.0,
    "EXIT_SHORT": 0.0,
    "HOLD": 0.0,
}

# --- WALK-FORWARD OPTIMIZATION (WFO) SETTINGS ---
WFO_SETTINGS = {
    "TRAINING_WINDOW_MONTHS": 12,
    "VALIDATION_WINDOW_MONTHS": 3,
    "STEP_SIZE_MONTHS": 3,
}

# --- DATABASE SETTINGS ---
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
DB_POOL_RECYCLE_SECONDS = 1800  # Recycle connections every 30 minutes

# --- MONITORING THRESHOLDS ---
MAX_CPU_PERCENT = 90.0
MAX_RAM_PERCENT = 90.0
MAX_GPU_TEMPERATURE = 85.0  # In Celsius
DATA_FRESHNESS_THRESHOLD_SECONDS = 1800  # 30 minutes

# --- SCHEDULER SETTINGS ---
API_FETCH_DELAY_SECONDS = 5  # Number of seconds past the minute to run API fetch jobs
