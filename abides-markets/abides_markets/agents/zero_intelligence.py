import logging
from typing import Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ..generators import OrderSizeGenerator
from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent


logger = logging.getLogger(__name__)


class ZeroIntelligence(TradingAgent):
    """
    Zero Intelligence agent implements a simple trading strategy.
    The agent wakes up periodically at regular intervals and places 1 order each time.
    """

    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "IBM",
        starting_cash: int = 100000,
        log_orders: bool = False,
        order_size_model: Optional[OrderSizeGenerator] = None,
        wakeup_time: Optional[NanosecondTime] = None,
        wake_up_interval: NanosecondTime = str_to_ns("15s"),
        price_std: float = 50.0,
        order_size_min: int = 20,
        order_size_max: int = 50,
        price_model: str = "lognormal",
    ) -> None:

        # Base class init.
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.wakeup_time: NanosecondTime = wakeup_time
        self.wake_up_interval: NanosecondTime = wake_up_interval

        self.symbol: str = symbol  # symbol to trade

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading: bool = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time: Optional[NanosecondTime] = None

        self.order_size: Optional[int] = (
            self.random_state.randint(order_size_min, order_size_max + 1) if order_size_model is None else None
        )

        self.order_size_model = order_size_model  # Probabilistic model for order size
        self.price_std: float = price_std
        self.price_model: str = price_model  # "lognormal" or "normal"

        # Remember last known bid and ask for price calculations
        self.last_bid: Optional[int] = None
        self.last_ask: Optional[int] = None
        self.order_size_min: int = order_size_min
        self.order_size_max: int = order_size_max

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

        self.oracle = self.kernel.oracle

    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

        # Fix the problem of logging an agent that has not waken up
        try:
            # noise trader surplus is marked to EOD
            bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)
        except KeyError:
            self.logEvent("FINAL_VALUATION", self.starting_cash, True)
        else:
            # Print end of day valuation.
            H = int(round(self.get_holdings(self.symbol), -2) / 100)

            if bid and ask:
                rT = int(bid + ask) / 2
            else:
                rT = self.last_trade[self.symbol]

            # final (real) fundamental value times shares held.
            surplus = rT * H

            logger.debug("Surplus after holdings: {}", surplus)

            # Add ending cash value and subtract starting cash value.
            surplus += self.holdings["CASH"] - self.starting_cash
            surplus = float(surplus) / self.starting_cash

            self.logEvent("FINAL_VALUATION", surplus, True)

            logger.debug(
                "{} final report.  Holdings: {}, end cash: {}, start cash: {}, final fundamental: {}, surplus: {}",
                self.name,
                H,
                self.holdings["CASH"],
                self.starting_cash,
                rT,
                surplus,
            )

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(current_time)

        self.state = "INACTIVE"

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                logger.debug("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        if self.wakeup_time > current_time:
            self.set_wakeup(self.wakeup_time)
            return

        if self.mkt_closed and self.symbol not in self.daily_close_price:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        if type(self) == ZeroIntelligence:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self.state = "ACTIVE"

    def compute_price_lognormal(self, mid_price: float) -> int:
        log_std = np.log1p(self.price_std)  # stable for small std
        log_return = -0.5 * log_std**2 + self.random_state.normal(0, log_std)
        price = max(1, int(np.rint(mid_price * np.exp(log_return))))
        return price

    def compute_price_normal(self, mid_price: float) -> int:
        price_noise = self.random_state.normal(0, mid_price * self.price_std)
        price = max(1, int(np.rint(mid_price + price_noise)))
        return price

    def place_orders(self) -> None:
        buy_indicator = self.random_state.randint(0, 1 + 1)

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        # Update last known bid/ask if available
        if bid:
            self.last_bid = bid
        if ask:
            self.last_ask = ask

        # Check if we have at least one price to work with
        if self.last_bid is None and self.last_ask is None:
            logger.warning(f"Agent {self.id}: No bid or ask available yet, cannot place order")
            return

        # Compute mid price from last known values
        if self.last_bid and self.last_ask:
            mid_price = (self.last_bid + self.last_ask) / 2
        elif self.last_bid:
            logger.warning(f"Agent {self.id}: No ask available, using last_bid={self.last_bid}")
            mid_price = self.last_bid
        else:  # only last_ask
            logger.warning(f"Agent {self.id}: No bid available, using last_ask={self.last_ask}")
            mid_price = self.last_ask

        if self.order_size_model is not None:
            self.order_size = self.order_size_model.sample(random_state=self.random_state)

        if self.order_size > 0:
            if self.price_model == "lognormal":
                price = self.compute_price_lognormal(mid_price)
            elif self.price_model == "normal":
                price = self.compute_price_normal(mid_price)
            else:
                raise ValueError(f"Unknown price_model: {self.price_model}")

            if buy_indicator == 1:
                self.place_limit_order(self.symbol, self.order_size, Side.BID, price)
            else:
                self.place_limit_order(self.symbol, self.order_size, Side.ASK, price)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receive_message(current_time, sender_id, message)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == "AWAITING_SPREAD":
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if isinstance(message, QuerySpreadResponseMsg):
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed:
                    return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                # Cancel all outstanding orders before placing new one
                self.cancel_all_orders()
                self.place_orders()

                # Schedule next wakeup
                self.set_wakeup(current_time + self.get_wake_frequency())
                self.state = "AWAITING_WAKEUP"

    # Internal state and logic specific to this agent subclass.

    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_up_interval
