from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Union

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ..messages.marketdata import MarketDataMsg, L2SubReqMsg
from ..orders import Side
from .trading_agent import TradingAgent

logger = logging.getLogger(__name__)


class TrendOrderSizeDistribution(ABC):

    @abstractmethod
    def sample(
        self, short_ma: float, long_ma: float, random_state: np.random.RandomState
    ) -> int:
        pass


class ProportionalOrderSize(TrendOrderSizeDistribution):

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def sample(
        self, short_ma: float, long_ma: float, random_state: np.random.RandomState
    ) -> int:
        ma_difference = abs(short_ma - long_ma)
        order_size = int(self.factor * ma_difference)
        return max(order_size, 1)


class TradeArrivalProcess(ABC):

    @abstractmethod
    def sample(self, random_state: np.random.RandomState) -> NanosecondTime:
        pass


class PoissonArrivalProcess(TradeArrivalProcess):

    def __init__(self, mean_interval: NanosecondTime):
        self.mean_interval = mean_interval

    def sample(self, random_state: np.random.RandomState) -> NanosecondTime:
        return int(random_state.exponential(scale=self.mean_interval))


class TrendFollowingAgent(TradingAgent):
    """
    Trend Following Agent that places orders based on the trend of the mid price.

    Args:
        price_offset: is the percentage offset from the mid price to place the order.
        threshold: is the threshold trigger for detecting the trend.
        sampling_freq: is the frequency of market data samples (candle period).
        trade_arrival: is the interval or process for placing orders.
        short_window: is the window size for the short moving average.
        long_window: is the window size for the long moving average.

    Overridable methods for custom moving average implementations:
        update_price_history(mid_price): Update internal state with new price data.
        compute_moving_average(window): Compute MA for given window size.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        sampling_freq: NanosecondTime = str_to_ns("300s"),
        trade_arrival: Union[NanosecondTime, TradeArrivalProcess] = str_to_ns("60min"),
        first_trade_time: Optional[NanosecondTime] = None,
        short_window: int = 20,
        long_window: int = 50,
        threshold: float = 0.01,
        order_size_model: Optional[TrendOrderSizeDistribution] = None,
        price_offset: float = 0.05,
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.sampling_freq = sampling_freq
        self.trade_arrival = trade_arrival
        self.first_trade_time = first_trade_time
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold
        self.order_size_model = order_size_model
        self.price_offset = price_offset

        self.mid_prices: List[float] = []
        self.next_trade_time: Optional[NanosecondTime] = None
        self.subscribed = False

    def wakeup(self, current_time: NanosecondTime) -> None:
        can_trade = super().wakeup(current_time)

        if can_trade and not self.subscribed:
            super().request_data_subscription(
                L2SubReqMsg(symbol=self.symbol, freq=self.sampling_freq, depth=1)
            )
            self.subscribed = True

            if self.first_trade_time is not None:
                self.next_trade_time = self.first_trade_time
            else:
                self.next_trade_time = current_time

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)

        if isinstance(message, MarketDataMsg):
            self.cancel_all_orders()

            bids = self.known_bids[self.symbol]
            asks = self.known_asks[self.symbol]

            if bids and asks:
                bid = bids[0][0]
                ask = asks[0][0]
                mid_price = (bid + ask) / 2

                self.update_price_history(mid_price)
                self.logEvent("BID_ASK", {"bid": bid, "ask": ask, "mid": mid_price})

                if self.time_to_trade(current_time):
                    self.execute_trend_strategy(mid_price, current_time)
                    self.schedule_next_trade(current_time)

    def time_to_trade(self, current_time: NanosecondTime) -> bool:
        if self.next_trade_time is None:
            return False
        return current_time >= self.next_trade_time

    def schedule_next_trade(self, current_time: NanosecondTime) -> None:
        if isinstance(self.trade_arrival, TradeArrivalProcess):
            next_interval = self.trade_arrival.sample(self.random_state)
        else:
            next_interval = self.trade_arrival

        self.next_trade_time = current_time + next_interval

    def execute_trend_strategy(
        self, mid_price: float, current_time: NanosecondTime
    ) -> None:
        if len(self.mid_prices) < self.long_window:
            return

        short_ma = self.compute_moving_average(self.short_window)
        long_ma = self.compute_moving_average(self.long_window)

        if long_ma == 0:
            return

        ma_difference = (short_ma - long_ma) / long_ma

        if abs(ma_difference) < self.threshold:
            return

        order_size = self.get_order_size(short_ma, long_ma)
        if order_size <= 0:
            return

        if ma_difference > self.threshold:
            limit_price = int(mid_price * (1 + self.price_offset))
            # logger.info(f"Placing BUY order of size {order_size} @ price {limit_price}")
            self.place_limit_order(
                self.symbol, quantity=order_size, side=Side.BID, limit_price=limit_price
            )
        elif ma_difference < -self.threshold:
            limit_price = int(mid_price * (1 - self.price_offset))
            # logger.info(f"Placing SELL order of size {order_size} @ price {limit_price}")
            self.place_limit_order(
                self.symbol, quantity=order_size, side=Side.ASK, limit_price=limit_price
            )

    def close_position(self, mid_price: float, position: int) -> None:
        quantity = abs(position)

        if position > 0:
            limit_price = int(mid_price * (1 - self.price_offset))
            # logger.info(f"Taking profit: SELL {quantity} @ price {limit_price}")
            self.place_limit_order(
                self.symbol, quantity=quantity, side=Side.ASK, limit_price=limit_price
            )
        elif position < 0:
            limit_price = int(mid_price * (1 + self.price_offset))
            # logger.info(f"Taking profit: BUY {quantity} @ price {limit_price}")
            self.place_limit_order(
                self.symbol, quantity=quantity, side=Side.BID, limit_price=limit_price
            )

    def update_price_history(self, mid_price: float) -> None:
        self.mid_prices.append(mid_price)
        if len(self.mid_prices) > self.long_window:
            self.mid_prices.pop(0)

    def compute_moving_average(self, window: int) -> float:
        recent_prices = self.mid_prices[-window:]
        return sum(recent_prices) / len(recent_prices)

    def get_order_size(self, short_ma: float, long_ma: float) -> int:
        if self.order_size_model is not None:
            return self.order_size_model.sample(short_ma, long_ma, self.random_state)
        return 100

    def get_wake_frequency(self) -> NanosecondTime:
        return str_to_ns("1s")


class ExponentialTrendFollowingAgent(TrendFollowingAgent):
    """
    Trend Following Agent using Exponential Moving Averages (EMA).

    Overrides price history management to use EMAs instead of simple moving averages.
    Memory efficient: stores only current EMA values, not historical prices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.short_ema: Optional[float] = None
        self.long_ema: Optional[float] = None
        self.alpha_short = 2 / (self.short_window + 1)
        self.alpha_long = 2 / (self.long_window + 1)

    def update_price_history(self, mid_price: float) -> None:
        if self.short_ema is None:
            self.short_ema = mid_price
            self.long_ema = mid_price
        else:
            self.short_ema = self.alpha_short * mid_price + (1 - self.alpha_short) * self.short_ema
            self.long_ema = self.alpha_long * mid_price + (1 - self.alpha_long) * self.long_ema

    def compute_moving_average(self, window: int) -> float:
        if window == self.short_window:
            return self.short_ema if self.short_ema is not None else 0.0
        elif window == self.long_window:
            return self.long_ema if self.long_ema is not None else 0.0
        else:
            raise ValueError(f"Unknown window size: {window}")
