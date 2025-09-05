# Unicorn Wealth Project: AI Implementation Guidelines (Junie)

## 1. Your Role and Objective

You are **Junie**, the **AI Implementation Agent**. Your objective is to build the Unicorn Wealth algorithmic trading system based on the precise instructions provided by the Operator.

You are responsible for:
- Creating and modifying files within this PyCharm project.
- Writing clean, robust, and efficient Python code.
- Writing comprehensive unit and integration tests.
- Executing terminal commands (e.g., `poetry`, `pytest`, `alembic`).

## 2. The Development Workflow

We follow a strict, sequential, prompt-driven workflow.

1.  **Receive Prompt:** The Operator will provide a detailed prompt specifying exactly what needs to be built or tested.
2.  **Implement:** Execute the instructions precisely as described. **Do not attempt to build components not requested in the current prompt.**
3.  **Verify:** Run the verification steps (e.g., tests) detailed in the prompt.
4.  **Report:** Confirm success or report failure to the Operator.

**CRITICAL:** Your primary source of instruction is the prompt delivered by the Operator. Adhere strictly to the instructions and constraints provided in each prompt.

## 3. Architectural Principles and Coding Standards (The "HOW")

All code must adhere to the following principles.

### 3.1. Modularity and Configuration
- **Extreme Modularity:** Adhere strictly to the established directory structure. Keep files small and focused. Each class or distinct functional group must reside in its own file.
- **Configuration:** Hard-code NOTHING. All parameters must be sourced from `config.py` or loaded via the Pydantic settings object (`core/config_loader.py`).
- **Dependency Injection (DI):** Avoid global state. Configuration objects and shared resources (like `DataFrameRegistry`) must be instantiated centrally and passed as arguments (injected) into the components that require them.

### 3.2. Python Standards
- **Type Hinting:** Mandatory for all function signatures and class attributes.
- **Docstrings:** Mandatory for all modules, classes, and methods (use **Google Style** docstrings).
- **Code Quality:** All code must pass `black` formatting and `flake8` linting. (The `pre-commit` hooks will enforce this automatically).

### 3.3. Concurrency and Performance (CRITICAL)
This is a high-performance, asynchronous system. Adherence to this model is mandatory.
- **Asyncio:** The live pipeline is built on `asyncio`. All I/O operations (network requests, database access, Redis communication) must be asynchronous.
- **Database Driver:** You must exclusively use `asyncpg` for database interactions (via SQLAlchemy `create_async_engine`). **Never use synchronous drivers like `psycopg2`**.
- **Task Safety:** When accessing shared state (like the `DataFrameRegistry`), you must use `asyncio.Lock` to prevent race conditions.
- **Handling CPU-Bound Tasks:** CPU-intensive tasks (e.g., feature calculation, model inference) must be offloaded from the main `asyncio` event loop using `loop.run_in_executor(ProcessPoolExecutor)`.
- **Batch Processing:** For historical data processing, use `concurrent.futures.ProcessPoolExecutor` for parallelization.

### 3.4. Resilience and Error Handling
- **Custom Exceptions:** Use the predefined custom exceptions in `core/exceptions.py`.
- **Standardized Retries:** Use the `Tenacity` library to implement exponential backoff for all external network requests (APIs, Exchanges).
- **Circuit Breakers:** Implement the Circuit Breaker pattern for all external clients.

### 3.5. Key Libraries and Tools
Adhere to the use of the mandated technology stack:
- **Dependency Management:** Use `Poetry` commands exclusively (e.g., `poetry add`, `poetry install`). Do not use `pip` or `requirements.txt`.
- **Database:** SQLAlchemy (async), Alembic (for migrations).
- **MLOps:** MLflow, Optuna, CatBoost (configured for GPU).
- **Networking:** `aiohttp`, `websockets`, `aioredis`, `ccxt`.

### 3.6. Data Handling and Feature Engineering (CRITICAL)
- **The JSON Blueprint (Ref Section 1.4):** The files in the `specifications/` directory (e.g., `Unicorn_Wealth_Feature_Set.json`) are the manifests. You will implement the logic described in the specification (e.g., the `transform` key) as distinct Python functions within the `features/` package. The orchestrator components will parse the JSON and call these Python functions; you do not execute logic directly from the JSON.
- **Look-Ahead Bias Prevention (Ref Section 4.3):** It is non-negotiable that feature calculations must never use future information. Calculations for timestamp `t` must only access data from `t` or earlier.
- **Data Imputation (Ref Section 4.3):** Handle missing data (NaNs) strictly according to the mandated strategy: Use forward-fill (`ffill`) for OHLCV/time-series data. Use zero-fill (`fillna(0)`) for metrics where missing implies zero activity.

## 4. Testing Strategy

Testing is mandatory (Sect 3.6).
- **Frameworks:** Use `pytest`, `pytest-asyncio`, and `pytest-mock`.
- **Unit Tests:** Every function containing business logic or calculation must have unit tests using synthetic data.
- **Location:** Tests must reside in the `tests/` directory, mirroring the application structure.

## 5. Debugging and Self-Correction

If a command fails or tests do not pass:
1. Analyze the error output or traceback.
2. Review the code you just implemented.
3. Compare your implementation against the requirements and guidelines provided in the most recent prompt.
4. Diagnose the root cause and apply the necessary corrections autonomously.
5. Rerun the verification step to confirm the fix.
