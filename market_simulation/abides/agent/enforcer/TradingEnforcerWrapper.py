import ctypes
import os

# dtimer_t is uint64_t in C
dtimer_t = ctypes.c_uint64

class Inputs_TradingEnforcer(ctypes.Structure):
    _fields_ = [
        ("dd_exceeded",       ctypes.c_bool),
        ("is_illiquid",       ctypes.c_bool),
        ("will_exceed_limit", ctypes.c_bool),
        ("price_deviates",    ctypes.c_bool),
        ("act_EXEC",          ctypes.c_bool),
    ]


class Outputs_TradingEnforcer(ctypes.Structure):
    _fields_ = [
        ("act_PLACE",  ctypes.c_bool),
        ("act_BUY",    ctypes.c_bool),
        ("act_MSG",    ctypes.c_bool),
        ("act_CANCEL", ctypes.c_bool),
        ("act_SELL",   ctypes.c_bool),
        ("price",      ctypes.c_int32),
    ]


class EnforcerVars_TradingEnforcer(ctypes.Structure):
    """Must mirror enforcervars_TradingEnforcer_t field‑by‑field."""
    _fields_ = [
        # RateLimit_5_per_1s
        ("_policy_RateLimit_5_per_1s_state", ctypes.c_int),
        ("t_window", dtimer_t),

        # LatchingKillSwitch
        ("_policy_LatchingKillSwitch_state", ctypes.c_int),

        # RejectDeviantPrice
        ("_policy_RejectDeviantPrice_state", ctypes.c_int),

        # BlockConcentratedBuy
        ("_policy_BlockConcentratedBuy_state", ctypes.c_int),

        # BlockIlliquidTrade
        ("_policy_BlockIlliquidTrade_state", ctypes.c_int),

        # OTR_Policy
        ("_policy_OTR_Policy_state", ctypes.c_int),
        ("tokens", ctypes.c_int32),

        # MQL_Policy
        ("_policy_MQL_Policy_state", ctypes.c_int),
        ("c_age", dtimer_t),

        # NoWash
        ("_policy_NoWash_state", ctypes.c_int),
        ("last_bid", ctypes.c_int32),
        ("last_ask", ctypes.c_int32),
    ]


_DEFAULT_DRAWDOWN_PCT       = 0.10   # 10 % max drawdown from peak
_DEFAULT_LIQUIDITY_MIN      = 200    # minimum total shares on book
_DEFAULT_CONCENTRATION_PCT  = 0.25   # max 25 % of portfolio in one stock
_DEFAULT_PRICE_DEVIATION    = 0.05   # 5 % band around mid‑price


class TradingEnforcer:

    HOLD   = 0
    BUY    = 1
    SELL   = 2
    CANCEL = 3

    def __init__(
        self,
        lib_path: str = "./libtradingenforcer.so",
        starting_cash: float = 10000000,
        drawdown_pct: float = _DEFAULT_DRAWDOWN_PCT,
        liquidity_min: int = _DEFAULT_LIQUIDITY_MIN,
        concentration_pct: float = _DEFAULT_CONCENTRATION_PCT,
        price_deviation_pct: float = _DEFAULT_PRICE_DEVIATION,
    ):
        # --- Load shared library ---
        abs_path = os.path.abspath(lib_path)
        try:
            self.lib = ctypes.CDLL(abs_path)
        except OSError as e:
            raise RuntimeError(
                f"Could not load TradingEnforcer library at {abs_path}: {e}"
            )

        # --- Declare C function signatures ---
        self.lib.TradingEnforcer_init_all_vars.argtypes = [
            ctypes.POINTER(EnforcerVars_TradingEnforcer),
            ctypes.POINTER(Inputs_TradingEnforcer),
            ctypes.POINTER(Outputs_TradingEnforcer),
        ]

        self.lib.TradingEnforcer_run_via_enforcer.argtypes = [
            ctypes.POINTER(EnforcerVars_TradingEnforcer),
            ctypes.POINTER(Inputs_TradingEnforcer),
            ctypes.POINTER(Outputs_TradingEnforcer),
        ]

        # --- Allocate & initialise C structs ---
        self.enf_vars = EnforcerVars_TradingEnforcer()
        self.inputs   = Inputs_TradingEnforcer()
        self.outputs  = Outputs_TradingEnforcer()

        self.lib.TradingEnforcer_init_all_vars(
            ctypes.byref(self.enf_vars),
            ctypes.byref(self.inputs),
            ctypes.byref(self.outputs),
        )

        # --- Threshold parameters ---
        self.base_portfolio_value = starting_cash
        self.drawdown_pct         = drawdown_pct
        self.liquidity_min        = liquidity_min
        self.concentration_pct    = concentration_pct
        self.price_deviation_pct  = price_deviation_pct

    # ------------------------------------------------------------------
    # Core: evaluate inputs → call enforcer → return corrected action
    # ------------------------------------------------------------------

    def validate(
        self,
        action: int,
        bid_price: int,
        ask_price: int,
        portfolio_value: float,
        liquidity: int,
        stock_price: int,
        trade_executed: bool = False
    ) -> int:
        """
        Translate the RL agent's proposed *action* and current market
        context into the enforcer's boolean input/output interface, run the
        compiled enforcer, and return the (possibly overridden) action.

        Parameters
        ----------
        action : int
            0 = HOLD, 1 = BUY, 2 = SELL, 3 = CANCEL
        portfolio_value : float
            Current mark‑to‑market portfolio value.
        liquidity : int
            Total visible depth (bid + ask shares) on the order book.
        stock_price : int
            Current mid‑price (in cents, matching ABIDES convention).

        Returns
        -------
        int  –  The enforced action (may differ from the proposed one).
        """

        # ---------------------------------------------------------------
        # 1.  Compute boolean INPUT signals from raw market data
        # ---------------------------------------------------------------

        # -- Drawdown exceeded? --
        drawdown = (
            (self.base_portfolio_value - portfolio_value) / self.base_portfolio_value
            if self.base_portfolio_value > 0
            else 0.0
        )
        dd_exceeded = drawdown >= self.drawdown_pct

        # -- Market illiquid? --
        is_illiquid = liquidity < self.liquidity_min
        
        will_exceed_limit = False  # placeholder – implement as needed based on concentration_pct

        # -- Price deviates too far? --
        #    For a HOLD or CANCEL there is no price to check.
        price_deviates = False
        if(action == self.BUY):
            price_deviates = (abs(bid_price - stock_price) / stock_price) > self.price_deviation_pct
        elif(action == self.SELL):
            price_deviates = (abs(ask_price - stock_price) / stock_price) > self.price_deviation_pct
            

        # ---------------------------------------------------------------
        # 2.  Pack INPUT struct
        # ---------------------------------------------------------------
        self.inputs.dd_exceeded       = dd_exceeded
        self.inputs.is_illiquid       = is_illiquid
        self.inputs.will_exceed_limit = will_exceed_limit
        self.inputs.price_deviates    = price_deviates
        self.inputs.act_EXEC          = trade_executed

        # ---------------------------------------------------------------
        # 3.  Pack proposed OUTPUT struct  (what the agent *wants* to do)
        # ---------------------------------------------------------------
        self.outputs.act_PLACE  = action in (self.BUY, self.SELL)
        self.outputs.act_BUY    = action == self.BUY
        self.outputs.act_SELL   = action == self.SELL
        self.outputs.act_CANCEL = action == self.CANCEL
        self.outputs.act_MSG    = action != self.HOLD  # any non‑hold is a message
        self.outputs.price      = bid_price if action == self.BUY else ask_price if action == self.SELL else 0

        # ---------------------------------------------------------------
        # 4.  Run the compiled enforcer (all 8 policies at once)
        # ---------------------------------------------------------------
        self.lib.TradingEnforcer_run_via_enforcer(
            ctypes.byref(self.enf_vars),
            ctypes.byref(self.inputs),
            ctypes.byref(self.outputs),
        )

        # ---------------------------------------------------------------
        # 5.  Decode the enforced outputs back into an action integer
        # ---------------------------------------------------------------
        enforced_action = self._decode_action()
        return enforced_action
    

    # Directly set all inputs/outputs and run the enforcer (for testing)
    def run_via_enforcer(self,
                         dd_exceeded: bool = False,
                         is_illiquid: bool = False,
                         will_exceed_limit: bool = False,
                         price_deviates: bool = False,
                         act_EXEC: bool = False,
                         act_PLACE: bool = False,
                         act_BUY: bool = False,
                         act_MSG: bool = False,
                         act_CANCEL: bool = False,
                         act_SELL: bool = False,
                         price: int = 0
                         ) -> int:
    
        # Setting inputs
        self.inputs.dd_exceeded = dd_exceeded
        self.inputs.is_illiquid = is_illiquid
        self.inputs.will_exceed_limit = will_exceed_limit
        self.inputs.price_deviates = price_deviates
        self.inputs.act_EXEC = act_EXEC

        # Setting outputs
        self.outputs.act_PLACE = act_PLACE
        self.outputs.act_BUY = act_BUY
        self.outputs.act_MSG = act_MSG
        self.outputs.act_CANCEL = act_CANCEL
        self.outputs.act_SELL = act_SELL
        self.outputs.price = price

        # Run the enforcer
        self.lib.TradingEnforcer_run_via_enforcer(
            ctypes.byref(self.enf_vars),
            ctypes.byref(self.inputs),
            ctypes.byref(self.outputs),
        )

        # Decode and return the enforced action
        enforced_action = self._decode_action()
        return enforced_action

    def _decode_action(self) -> int:
        """Convert the (possibly modified) output booleans back to an int."""
        if self.outputs.act_BUY and self.outputs.act_PLACE and self.outputs.act_MSG:
            return self.BUY
        if self.outputs.act_SELL and self.outputs.act_PLACE and self.outputs.act_MSG:
            return self.SELL
        if self.outputs.act_CANCEL and self.outputs.act_MSG:
            return self.CANCEL
        return self.HOLD
