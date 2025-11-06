# Market Simulation Configuration)

import time
import numpy as np

from abides_core.utils import str_to_ns, datetime_str_to_ns
from abides_markets.generators import UniformOrderSizeGenerator
from abides_markets.agents import (
    ExchangeAgent,
    LiquidityMarketMakerAgent,
    SymmetricHumpLiquidityModel,
)
from abides_markets.agents.zero_intelligence import ZeroIntelligence as ZeroIntelligenceAgent, LogNormalPriceDistribution
from abides_markets.agents.trend_following_agent import TrendContrarianAgent, TrendFollowingAgent, ProportionalOrderSize, PoissonArrivalProcess
from abides_markets.orders import Side, LimitOrder
from abides_markets.utils import generate_latency_model
from abides_markets.order_book import OrderBook


# Trading precision constants
LOT_SIZE = 100_000  # 1 real stock = 100,000 quantity units (1 unit = 0.00001 real stocks)
TICK_SIZE = 100_000  # Price expressed in 1/100_000 of dollar (1 price unit = $0.00001)

# Initial order book parameters
REAL_STOCK_PRICE = 100_000  # Real stock price in dollars
INITIAL_PRICE = REAL_STOCK_PRICE * TICK_SIZE // LOT_SIZE
INITIAL_VOLUME = LOT_SIZE  # 1 real stock total (0.5 per side)

# Zero Intelligence agent parameters
ZI_NB_AGENTS = 15  # Number of Zero Intelligence (noise) agents
ZI_PRICE_STD = 0.1 / 100.  # 0.1% standard deviation relative to mid price
ZI_ORDER_SIZE_MIN = int(1.0 * LOT_SIZE)  # 1.0 real stocks
ZI_ORDER_SIZE_MAX = int(5.0 * LOT_SIZE)  # 5.0 real stocks
ZI_WAKE_UP_INTERVAL = str_to_ns("30s")  # Wake up every 15 seconds

# Trend Following agent parameters
TF_NB_AGENTS = 1
TF_SHORT_WINDOW = 12
TF_LONG_WINDOW = 40
TF_THRESHOLD = 0.05 / 100.
TF_PRICE_OFFSET = 2.0 / 100 # Amplitude of a shock when the liquidity is low
TF_ORDER_SIZE_MODEL = ProportionalOrderSize(factor=500 * LOT_SIZE, boost=0.01 * LOT_SIZE)
TF_TRADE_ARRIVAL_INTERVAL = str_to_ns("5min")
TF_SAMPLING_FREQ = str_to_ns("60s")

# Mean reverting agent parameters
TC_NB_AGENTS = 1
TC_SHORT_WINDOW = 3
TC_LONG_WINDOW = 60
TC_THRESHOLD = 0.08 / 100
TC_PRICE_OFFSET = 2.5 / 100 # Amplitude of a shock when the liquidity is low
# Boost control the shocks
TC_ORDER_SIZE_MODEL = ProportionalOrderSize(factor=1000 * LOT_SIZE, boost= 0.01 * LOT_SIZE)
TC_TRADE_ARRIVAL_INTERVAL = str_to_ns("5min")
TC_SAMPLING_FREQ = str_to_ns("60s")

# Liquidity Market Maker parameters
LMM_NB_AGENTS = 0
LMM_TOTAL_LIQUIDITY = 10 * LOT_SIZE
LMM_STEP_SIZE_RATIO = 100.0 / 100_000
LMM_MAX_LEVELS = 100
LMM_IMBALANCE_BETA = 0.0
LMM_SAMPLING_FREQ = str_to_ns("10s")
LMM_TRADE_ARRIVAL_INTERVAL = str_to_ns("10s")
LMM_PEAK_DISTANCE_RATIO = 5_000.0 / 100_000
LMM_SHAPE_EXPONENT = 1.2

def populate_initial_order_book(order_book: OrderBook, mkt_open: int) -> None:
    """Populate order book with initial orders."""
    vol = INITIAL_VOLUME // 2
    initial_orders = [
        LimitOrder(0, mkt_open, order_book.symbol, vol, Side.BID, INITIAL_PRICE - 1),
        LimitOrder(0, mkt_open, order_book.symbol, vol, Side.ASK, INITIAL_PRICE + 1),
    ]
    for order in initial_orders:
        order_book.enter_order(order, quiet=True)

# General configuration
def build_config(
    ticker="ABM",
    historical_date="20250101",
    start_time="00:00:00",
    end_time=None,
    exchange_log_orders=True,
    log_orders=True,
    book_logging=True,
    book_log_depth=10,
    seed=int(time.time_ns()) % (2 ** 32 - 1),
    stdout_log_level="INFO",
    ##
    num_trend_following_agents=TF_NB_AGENTS,
    num_noise_agents=ZI_NB_AGENTS,
    ##
    num_trend_contrarian_agents=TC_NB_AGENTS,
    ## liquidity market maker
    num_market_makers=LMM_NB_AGENTS,
    mm_total_liquidity=LMM_TOTAL_LIQUIDITY,
    mm_step_size_ratio=LMM_STEP_SIZE_RATIO,
    mm_max_levels=LMM_MAX_LEVELS,
    mm_imbalance_beta=LMM_IMBALANCE_BETA,
    mm_sampling_freq=LMM_SAMPLING_FREQ,
    mm_trade_arrival_interval=LMM_TRADE_ARRIVAL_INTERVAL,
    mm_peak_distance_ratio=LMM_PEAK_DISTANCE_RATIO,
    mm_shape_exponent=LMM_SHAPE_EXPONENT,
):
    symbol = ticker

    ##setting numpy seed
    np.random.seed(seed)

    ########################################################################################################################
    ############################################### AGENTS CONFIG ##########################################################

    # Historical date to simulate.
    historical_date = datetime_str_to_ns(historical_date)
    mkt_open = historical_date + str_to_ns(start_time)
    # Determine simulation end time
    if end_time is None:
        simulation_end = mkt_open + str_to_ns("3d")  # Run for 3 days by default
    else:
        simulation_end = historical_date + str_to_ns(end_time)
    # Market closes after simulation ends
    mkt_close = simulation_end - str_to_ns("60s")
    agent_count, agents = 0, []

    # Hyperparameters
    starting_cash = 10000000  # Cash in this simulator is always in CENTS.

    # 1) Exchange Agent

    #  How many orders in the past to store for transacted volume computation
    agents.extend(
        [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
                mkt_open=mkt_open,
                mkt_close=mkt_close,
                symbols=[symbol],
                book_logging=book_logging,
                book_log_depth=book_log_depth,
                log_orders=exchange_log_orders,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=25_000,
            )
        ]
    )
    agent_count += 1

    # 2) Zero Intelligence Agents
    num_zi = num_noise_agents
    agents.extend(
        [
            ZeroIntelligenceAgent(
                id=j,
                symbol=symbol,
                wakeup_time=mkt_open + np.random.randint(0, ZI_WAKE_UP_INTERVAL + 1),
                wake_up_interval=ZI_WAKE_UP_INTERVAL,
                log_orders=log_orders,
                price_model=LogNormalPriceDistribution(ZI_PRICE_STD),
                order_size_model=UniformOrderSizeGenerator(ZI_ORDER_SIZE_MIN, ZI_ORDER_SIZE_MAX, np.random.RandomState(seed)),
            )
            for j in range(agent_count, agent_count + num_zi)
        ]
    )
    agent_count += num_zi

    # 3) Liquidity Market Maker Agents
    num_mm_agents = num_market_makers

    agents.extend(
        [
            LiquidityMarketMakerAgent(
                id=j,
                name="LIQUIDITY_MARKET_MAKER_AGENT_{}".format(j),
                type="LiquidityMarketMakerAgent",
                symbol=symbol,
                starting_cash=starting_cash,
                random_state=np.random.RandomState(seed + j),
                liquidity_model=SymmetricHumpLiquidityModel(
                    peak_distance_ratio=mm_peak_distance_ratio,
                    shape_exponent=mm_shape_exponent,
                ),
                total_liquidity=mm_total_liquidity,
                step_size_ratio=mm_step_size_ratio,
                max_levels=mm_max_levels,
                imbalance_beta=mm_imbalance_beta,
                sampling_freq=mm_sampling_freq,
                trade_arrival=mm_trade_arrival_interval,
                first_wake_time=mkt_open + np.random.randint(0, mm_trade_arrival_interval),
                log_orders=log_orders,
            )
            for j in range(agent_count, agent_count + num_mm_agents)
        ]
    )
    agent_count += num_mm_agents

    # 4) Trend Following Agents
    num_tf_agents = num_trend_following_agents

    agents.extend(
        [
            TrendFollowingAgent(
                id=j,
                name="TREND_FOLLOWING_AGENT_{}".format(j),
                symbol=symbol,
                starting_cash=starting_cash,
                random_state=np.random.RandomState(seed + j),
                sampling_freq=TF_SAMPLING_FREQ,
                trade_arrival=PoissonArrivalProcess(TF_TRADE_ARRIVAL_INTERVAL),
                first_trade_time=mkt_open + np.random.randint(0, TF_TRADE_ARRIVAL_INTERVAL),
                short_window=TF_SHORT_WINDOW,
                long_window=TF_LONG_WINDOW,
                threshold=TF_THRESHOLD,
                order_size_model=TF_ORDER_SIZE_MODEL,
                price_offset=TF_PRICE_OFFSET,
                log_orders=log_orders,
            )
            for j in range(agent_count, agent_count + num_tf_agents)
        ]
    )
    agent_count += num_tf_agents

    # 5) Mean reversion Agents
    num_tf_agents = num_trend_contrarian_agents

    agents.extend(
        [
            TrendContrarianAgent(
                id=j,
                name="TREND_CONTRARIAN_AGENT_{}".format(j),
                symbol=symbol,
                starting_cash=starting_cash,
                random_state=np.random.RandomState(seed + j),
                sampling_freq=TC_SAMPLING_FREQ,
                trade_arrival=PoissonArrivalProcess(TC_TRADE_ARRIVAL_INTERVAL),
                first_trade_time=mkt_open + np.random.randint(0, TC_TRADE_ARRIVAL_INTERVAL),
                short_window=TC_SHORT_WINDOW,
                long_window=TC_LONG_WINDOW,
                threshold=TC_THRESHOLD,
                order_size_model=TC_ORDER_SIZE_MODEL,
                price_offset=TC_PRICE_OFFSET,
                log_orders=log_orders,
            )
            for j in range(agent_count, agent_count + num_tf_agents)
        ]
    )
    agent_count += num_tf_agents

    # LATENCY

    latency_model = generate_latency_model(agent_count)
    default_computation_delay = 50  # 50 nanoseconds

    # Populate initial order book
    exchange_agent = agents[0]
    for sym in exchange_agent.order_books:
        populate_initial_order_book(exchange_agent.order_books[sym], mkt_open)

    ##kernel args
    kernelStartTime = historical_date
    kernelStopTime = simulation_end

    return {
        "start_time": kernelStartTime,
        "stop_time": kernelStopTime,
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": default_computation_delay,
        "custom_properties": {"oracle": None},
        "stdout_log_level": stdout_log_level,
    }
