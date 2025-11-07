from abc import ABC, abstractmethod
import logging
from typing import Optional, Union

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ...messages.marketdata import MarketDataMsg, L2SubReqMsg
from ...orders import Side
from ..trading_agent import TradingAgent
from ..trend_following_agent import TradeArrivalProcess

logger = logging.getLogger(__name__)


def round_to_tick(price: float, tick: float, side: Side) -> int:
    """Round price to tick size, ensuring bids never cross up and asks never cross down."""
    if side == Side.BID:
        return int(tick * np.floor(price / tick))
    else:  # Side.ASK
        return int(tick * np.ceil(price / tick))


class LiquidityModel(ABC):
    """Abstract base class for liquidity distribution models."""

    @abstractmethod
    def compute_liquidity(
        self, price: np.ndarray, mid_price: float, imbalance: float
    ) -> np.ndarray:
        """
        Compute liquidity mass at given price levels.

        Args:
            price: Array of price levels to evaluate
            mid_price: Current mid price
            imbalance: Inventory imbalance parameter (-1 to 1)
                      0 = symmetric, >0 = more on ask side, <0 = more on bid side

        Returns:
            Array of liquidity masses at each price level
        """
        pass


class SymmetricHumpLiquidityModel(LiquidityModel):
    """
    Liquidity model based on a symmetric hump in log-price space.

    The model creates a distribution that:
    - Peaks at a specified distance from the mid-price
    - Decays exponentially as you move away from the peak
    - Can be skewed based on inventory imbalance
    - Works in log space for price invariance

    Args:
        peak_distance_ratio: Distance from mid-price where liquidity peaks (as fraction of mid-price)
        shape_exponent: Controls the sharpness of the hump (higher = sharper peak)
        near_mid_smoothing_fraction: Prevents infinite liquidity exactly at mid-price
    """

    def __init__(
        self,
        peak_distance_ratio: float = 0.05,
        shape_exponent: float = 1.2,
        near_mid_smoothing_fraction: float = 4e-4,
    ):
        self.peak_distance_ratio = peak_distance_ratio
        self.shape_exponent = shape_exponent
        self.near_mid_smoothing_fraction = near_mid_smoothing_fraction

    def compute_liquidity(
        self, price: np.ndarray, mid_price: float, imbalance: float
    ) -> np.ndarray:
        """
        Compute liquidity mass using log-price symmetric hump.

        Formula:
            symmetric_hump = (log_distance + smoothing)^exponent * exp(-decay_rate * log_distance)
            side_multiplier = 1 + imbalance for asks, 1 - imbalance for bids
            liquidity = symmetric_hump * side_multiplier

        Args:
            price: Array of price levels
            mid_price: Current mid price
            imbalance: Inventory imbalance (-1 to 1)

        Returns:
            Liquidity mass at each price level
        """
        price = np.asarray(price, dtype=float)

        # Compute peak fraction in log space
        peak_fraction = np.log(1.0 + self.peak_distance_ratio)

        # Distance from mid-price in log space
        log_distance = np.abs(np.log(price / float(mid_price)))

        # Decay rate to ensure peak at the specified distance
        decay_rate = self.shape_exponent / (
            peak_fraction + self.near_mid_smoothing_fraction
        )

        # Symmetric hump: rises to peak, then decays exponentially
        symmetric_hump = (
            (log_distance + self.near_mid_smoothing_fraction) ** self.shape_exponent
            * np.exp(-decay_rate * log_distance)
        )

        # Apply inventory skew: more liquidity on one side to reduce inventory
        # If holding long (positive inventory), want to sell more: imbalance > 0 increases ask side
        # If holding short (negative inventory), want to buy more: imbalance < 0 increases bid side
        side_multiplier = np.where(
            price > mid_price,
            1.0 + imbalance,  # Ask side
            np.where(price < mid_price, 1.0 - imbalance, 1.0),  # Bid side
        )

        return symmetric_hump * side_multiplier


class LiquidityMarketMakerAgent(TradingAgent):

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        liquidity_model: LiquidityModel,
        total_liquidity: int,
        step_size_ratio: float,
        max_levels: int = 100,
        imbalance_beta: float = 0.0,
        sampling_freq: NanosecondTime = str_to_ns("10s"),
        trade_arrival: Union[NanosecondTime, TradeArrivalProcess] = str_to_ns("10s"),
        first_wake_time: Optional[NanosecondTime] = None,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol = symbol
        self.liquidity_model = liquidity_model
        self.total_liquidity = total_liquidity
        self.step_size_ratio = step_size_ratio
        self.max_levels = max_levels
        self.imbalance_beta = imbalance_beta
        self.sampling_freq = sampling_freq
        self.trade_arrival = trade_arrival
        self.first_wake_time = first_wake_time

        self.subscribed = False
        self.next_wake_time: Optional[NanosecondTime] = None

    def wakeup(self, current_time: NanosecondTime) -> None:
        can_trade = super().wakeup(current_time)

        if can_trade and not self.subscribed:
            super().request_data_subscription(
                L2SubReqMsg(symbol=self.symbol, freq=self.sampling_freq, depth=1)
            )
            self.subscribed = True
            self.next_wake_time = (
                self.first_wake_time if self.first_wake_time is not None else current_time
            )

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)

        if isinstance(message, MarketDataMsg):
            bids = self.known_bids[self.symbol]
            asks = self.known_asks[self.symbol]

            if bids and asks:
                mid_price = np.sqrt(float(bids[0][0] * asks[0][0]))

                if self.time_to_update(current_time):
                    self.cancel_all_orders()
                    imbalance = self.compute_imbalance()
                    if mid_price > 0:
                        self.place_liquidity_orders(mid_price, imbalance)
                    self.schedule_next_wake(current_time)

    def time_to_update(self, current_time: NanosecondTime) -> bool:
        if self.next_wake_time is None:
            return False
        return current_time >= self.next_wake_time

    def schedule_next_wake(self, current_time: NanosecondTime) -> None:
        if isinstance(self.trade_arrival, TradeArrivalProcess):
            next_interval = self.trade_arrival.sample(self.random_state)
        else:
            next_interval = self.trade_arrival
        self.next_wake_time = current_time + next_interval

    def compute_imbalance(self) -> float:
        if self.imbalance_beta == 0:
            return 0.0
        holdings = self.get_holdings(self.symbol)
        imbalance = np.tanh(holdings * self.imbalance_beta / self.total_liquidity)
        return float(np.clip(imbalance, -1.0, 1.0))

    def compute_liquidity_distribution(self, mid_price: float, imbalance: float):
        log_step = np.log1p(self.step_size_ratio)

        bid_log_prices = np.arange(-log_step, -log_step * (self.max_levels + 1), -log_step)
        ask_log_prices = np.arange(log_step, log_step * (self.max_levels + 1), log_step)

        bid_prices = mid_price * np.exp(bid_log_prices)
        ask_prices = mid_price * np.exp(ask_log_prices)

        all_prices = np.concatenate([bid_prices, ask_prices])
        liquidity = self.liquidity_model.compute_liquidity(all_prices, mid_price, imbalance)

        total_mass = np.sum(liquidity) * log_step
        if total_mass > 0:
            liquidity = liquidity * self.total_liquidity / total_mass

        bid_liquidity = liquidity[: len(bid_prices)]
        ask_liquidity = liquidity[len(bid_prices) :]

        # Optional: per-side normalization so rounding noise doesn’t skew totals
        mass_bid = np.sum(bid_liquidity) * log_step
        mass_ask = np.sum(ask_liquidity) * log_step
        if mass_bid > 0: bid_liquidity *= (0.5 * self.total_liquidity) / mass_bid
        if mass_ask > 0: ask_liquidity *= (0.5 * self.total_liquidity) / mass_ask

        return bid_prices, bid_liquidity, ask_prices, ask_liquidity
    
    def place_liquidity_orders(self, mid_price: float, imbalance: float) -> None:
        bid_prices, bid_qtys, ask_prices, ask_qtys = self.compute_liquidity_distribution(
            mid_price, imbalance
        )

        orders = []
        for price, qty in zip(bid_prices, bid_qtys):
            quantity = int(np.round(qty))
            if quantity > 0:
                px = round_to_tick(price, 1, Side.BID)
                orders.append(
                    self.create_limit_order(self.symbol, quantity, Side.BID, px)
                )

        for price, qty in zip(ask_prices, ask_qtys):
            quantity = int(np.round(qty))
            if quantity > 0:
                px = round_to_tick(price, 1, Side.ASK)
                orders.append(
                    self.create_limit_order(self.symbol, quantity, Side.ASK, px)
                )

        if orders:
            self.place_multiple_orders(orders)

    def get_wake_frequency(self) -> NanosecondTime:
        return self.sampling_freq
