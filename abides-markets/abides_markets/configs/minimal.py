# Market Simulation Configuration)

import numpy as np

from abides_core.utils import str_to_ns, datetime_str_to_ns, get_wake_time
from abides_markets.agents import (
    ExchangeAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
#    POVExecutionAgent,
)
from abides_markets.agents.zero_intelligence import ZeroIntelligence as ZeroIntelligenceAgent
POVExecutionAgent = None
from abides_markets.oracles import SparseMeanRevertingOracle
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
NUM_NOISE_AGENTS = 25  # Number of Zero Intelligence (noise) agents
ZI_PRICE_STD = 0.005  # 0.5% standard deviation relative to mid price
ZI_ORDER_SIZE_MIN = int(0.1 * LOT_SIZE)  # 0.1 real stocks
ZI_ORDER_SIZE_MAX = int(0.5 * LOT_SIZE)  # 0.5 real stocks
ZI_WAKE_UP_INTERVAL = str_to_ns("15s")  # Wake up every 15 seconds

def populate_initial_order_book(order_book: OrderBook, mkt_open: int) -> None:
    """Populate order book with initial orders."""
    vol = INITIAL_VOLUME // 2
    initial_orders = [
        LimitOrder(0, mkt_open, order_book.symbol, vol, Side.BID, INITIAL_PRICE - 1),
        LimitOrder(0, mkt_open, order_book.symbol, vol, Side.ASK, INITIAL_PRICE + 1),
    ]
    for order in initial_orders:
        order_book.enter_order(order, quiet=True)


########################################################################################################################
############################################### GENERAL CONFIG #########################################################


def build_config(
    ticker="ABM",
    historical_date="20250101",
    start_time="00:00:00",
    end_time=None,
    exchange_log_orders=True,
    log_orders=True,
    book_logging=True,
    book_log_depth=10,
    seed=int(NanosecondTime.now().timestamp() * 1000000) % (2 ** 32 - 1),
    stdout_log_level="INFO",
    ##
    num_momentum_agents=0,
    num_noise_agents=NUM_NOISE_AGENTS,
    num_value_agents=0,
    ## exec agent
    execution_agents=False,
    execution_pov=0.1,
    ## market maker
    num_market_makers=0,
    mm_pov=0.025,
    mm_window_size="adaptive",
    mm_min_order_size=1,
    mm_num_ticks=10,
    mm_wake_up_freq=str_to_ns("10s"),
    mm_skew_beta=0,
    mm_level_spacing=5,
    mm_spread_alpha=0.75,
    mm_backstop_quantity=50_000,
    ##fundamental/oracle
    fund_r_bar=100_000,
    fund_kappa=1.67e-16,
    fund_sigma_s=0,
    fund_vol=1e-3,  # Volatility of fundamental time series (std).
    fund_megashock_lambda_a=2.77778e-18,
    fund_megashock_mean=1000,
    fund_megashock_var=50_000,
    ##value agent
    val_r_bar=100_000,
    val_kappa=1.67e-15,
    val_vol=1e-8,
    val_lambda_a=7e-11,
):
    fund_sigma_n = fund_r_bar / 10
    val_sigma_n = val_r_bar / 10
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
    agent_count, agents, agent_types = 0, [], []

    # Hyperparameters
    starting_cash = 10000000  # Cash in this simulator is always in CENTS.

    # Oracle
    symbols = {
        symbol: {
            "r_bar": fund_r_bar,
            "kappa": fund_kappa,
            "sigma_s": fund_sigma_s,
            "fund_vol": fund_vol,
            "megashock_lambda_a": fund_megashock_lambda_a,
            "megashock_mean": fund_megashock_mean,
            "megashock_var": fund_megashock_var,
            "random_state": np.random.RandomState(
                seed=np.random.randint(low=0, high=2**32, dtype="uint64")
            ),
        }
    }

    oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

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
    agent_types.extend("ExchangeAgent")
    agent_count += 1

    # 2) Zero Intelligence Agents
    num_zi = num_noise_agents
    agents.extend(
        [
            ZeroIntelligenceAgent(
                id=j,
                symbol=symbol,
                starting_cash=starting_cash,
                wakeup_time=mkt_open + np.random.randint(0, ZI_WAKE_UP_INTERVAL + 1),
                wake_up_interval=ZI_WAKE_UP_INTERVAL,
                log_orders=log_orders,
                price_std=ZI_PRICE_STD,
                order_size_min=ZI_ORDER_SIZE_MIN,
                order_size_max=ZI_ORDER_SIZE_MAX,
            )
            for j in range(agent_count, agent_count + num_zi)
        ]
    )
    agent_count += num_zi
    agent_types.extend(["ZeroIntelligenceAgent"])

    # 3) Value Agents
    num_value = num_value_agents
    agents.extend(
        [
            ValueAgent(
                id=j,
                name="Value Agent {}".format(j),
                symbol=symbol,
                starting_cash=starting_cash,
                sigma_n=val_sigma_n,
                r_bar=val_r_bar,
                kappa=val_kappa,
                lambda_a=val_lambda_a,
                log_orders=log_orders,
            )
            for j in range(agent_count, agent_count + num_value)
        ]
    )
    agent_count += num_value
    agent_types.extend(["ValueAgent"])

    # 4) Market Maker Agents

    """
    window_size ==  Spread of market maker (in ticks) around the mid price
    pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
           the market maker places at each level
    num_ticks == Number of levels to place orders in around the spread
    wake_up_freq == How often the market maker wakes up
    
    """

    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    mm_params = num_market_makers * [
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size)
    ]

    num_mm_agents = len(mm_params)
    mm_cancel_limit_delay = 50  # 50 nanoseconds

    agents.extend(
        [
            AdaptiveMarketMakerAgent(
                id=j,
                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                type="AdaptivePOVMarketMakerAgent",
                symbol=symbol,
                starting_cash=starting_cash,
                pov=mm_params[idx][1],
                min_order_size=mm_params[idx][4],
                window_size=mm_params[idx][0],
                num_ticks=mm_params[idx][2],
                wake_up_freq=mm_params[idx][3],
                cancel_limit_delay=mm_cancel_limit_delay,
                skew_beta=mm_skew_beta,
                level_spacing=mm_level_spacing,
                spread_alpha=mm_spread_alpha,
                backstop_quantity=mm_backstop_quantity,
                log_orders=log_orders,
            )
            for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))
        ]
    )
    agent_count += num_mm_agents
    agent_types.extend("POVMarketMakerAgent")

    # 5) Momentum Agents
    num_momentum_agents = num_momentum_agents

    agents.extend(
        [
            MomentumAgent(
                id=j,
                name="MOMENTUM_AGENT_{}".format(j),
                symbol=symbol,
                starting_cash=starting_cash,
                min_size=1,
                max_size=10,
                wake_up_freq=str_to_ns("20s"),
                log_orders=log_orders,
            )
            for j in range(agent_count, agent_count + num_momentum_agents)
        ]
    )
    agent_count += num_momentum_agents
    agent_types.extend("MomentumAgent")

    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )
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
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": stdout_log_level,
    }
