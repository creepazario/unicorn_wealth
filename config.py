# --- TOKEN UNIVERSE ---
MASTER_TOKEN_LIST = {
    "AAVE": {"santiment_slug": "aave", "amberdata_symbol": "aave_usd"},
    "SHIB": {"santiment_slug": "shiba-inu", "amberdata_symbol": "shib_usd"},
    "UNI": {"santiment_slug": "uniswap", "amberdata_symbol": "uni_usd"},
    "MKR": {"santiment_slug": "maker", "amberdata_symbol": "mkr_usd"},
    "LDO": {"santiment_slug": "lido-dao", "amberdata_symbol": "ldo_usd"},
    "IMX": {"santiment_slug": "immutable-x", "amberdata_symbol": "imx_usd"},
    "MANA": {"santiment_slug": "decentraland", "amberdata_symbol": "mana_usd"},
    "SNX": {"santiment_slug": "synthetix-network-token", "amberdata_symbol": "snx_usd"},
    "SAND": {"santiment_slug": "the-sandbox", "amberdata_symbol": "sand_usd"},
    "AXS": {"santiment_slug": "axie-infinity", "amberdata_symbol": "axs_usd"},
    "BAT": {"santiment_slug": "basic-attention-token", "amberdata_symbol": "bat_usd"},
    "ETH": {"santiment_slug": "ethereum", "amberdata_symbol": "eth_usd"},
    "QNT": {"santiment_slug": "quant", "amberdata_symbol": "qnt_usd"},
    "COMP": {"santiment_slug": "compound", "amberdata_symbol": "comp_usd"},
    "CHZ": {"santiment_slug": "chiliz", "amberdata_symbol": "chz_usd"},
    "RAY": {"santiment_slug": "raydium", "amberdata_symbol": "ray_usd"},
    "PAXG": {"santiment_slug": "pax-gold", "amberdata_symbol": "paxg_usd"},
    "MATIC": {"santiment_slug": "matic-network", "amberdata_symbol": "matic_usd"},
    "RNDR": {"santiment_slug": "render", "amberdata_symbol": "rndr_usd"},
    "PEPE": {"santiment_slug": "pepe", "amberdata_symbol": "pepe_usd"},
    "INJ": {"santiment_slug": "injective-protocol", "amberdata_symbol": "inj_usd"},
    "ENS": {"santiment_slug": "ethereum-name-service", "amberdata_symbol": "ens_usd"},
    "GRT": {"santiment_slug": "the-graph", "amberdata_symbol": "gr_usd"},
    "BTC": {"santiment_slug": "bitcoin", "amberdata_symbol": "btc_usd"},
}

# --- FEATURE PARAMETERS (HORIZON-SPECIFIC) ---
# These will be dynamically populated later and updated by the operator.
FEATURE_PARAMS = {
    "1h": {
        "adx_15m": {"window": 30},
        "rsi_15m": {"window": 40},
    },
    "4h": {
        "adx_15m": {"window": 30},
        "rsi_15m": {"window": 40},
    },
    "8h": {
        "adx_15m": {"window": 30},
        "rsi_15m": {"window": 40},
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
LOG_LEVEL = "INFO"
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
API_CLIENT_SETTINGS = {
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
