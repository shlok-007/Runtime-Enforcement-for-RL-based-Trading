from agent.examples.ExampleExperimentalAgent import ExampleExperimentalAgentTemplate
from util.util import log_print
import pandas as pd
from collections import defaultdict
import pickle
import os

from agent.enforcer.TradingEnforcerWrapper import TradingEnforcer

class RLAgent(ExampleExperimentalAgentTemplate):
    def __init__(self, id, name, type, symbol, starting_cash, 
                 levels=1, subscription_freq=1000000000, log_orders=False, random_state=None,
                 wake_freq='10s', order_size=100, 
                 alpha=0.1, gamma=0.9, epsilon=0.2,
                 enable_enforcer=True):

        super().__init__(id, name, type, symbol, starting_cash, levels, subscription_freq, 
                         log_orders=log_orders, random_state=random_state)

        self.wake_freq = wake_freq
        self.order_size = order_size
        
        # RL Hyperparameters
        self.alpha = alpha       # Learning rate
        self.gamma = gamma       # Discount factor
        self.epsilon = epsilon   # Exploration rate
        
        # Action Space: 0=HOLD, 1=BUY, 2=SELL, 3=CANCEL
        self.actions = [0, 1, 2, 3]
        self.actions_map = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CANCEL"}
        self.q_table_path = f"q_table_{self.name}.pkl"
        if self.q_table_path and os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                self.q_table = pickle.load(f)
                log_print("Loaded existing Q-Table from {}", self.q_table_path)
        else:
            self.q_table = defaultdict(float)
            log_print("Initialized new, empty Q-Table.")
        
        # RL State Tracking
        self.last_state = None
        self.last_action = None
        self.last_portfolio_value = starting_cash
        self.mid_price_history = []

        self.enforcer_enabled = enable_enforcer

        if(self.enforcer_enabled):
            self.enforcer = TradingEnforcer(
                lib_path="/home/shlok/Runtime-Enforcement-for-RL-based-Trading/policies/TradingEnforcer/lib_TradingEnforcer_2.so",
                starting_cash=starting_cash,
            )

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_freq)

    def get_current_state(self):
        """ Defines the state space based on inventory and price momentum. """
        # 1. Inventory State (-1: Short, 0: Flat, 1: Long)
        holdings = self.getHoldings(self.symbol)
        if holdings > 0:
            inv_state = 1
        elif holdings < 0:
            inv_state = -1
        else:
            inv_state = 0
            
        # 2. Price Momentum State (-1: Down, 0: Flat, 1: Up)
        momentum_state = 0
        if len(self.mid_price_history) >= 2:
            current_mid = self.mid_price_history[-1]
            prev_mid = self.mid_price_history[-2]
            if current_mid > prev_mid:
                momentum_state = 1
            elif current_mid < prev_mid:
                momentum_state = -1
                
        return (inv_state, momentum_state)

    def choose_action(self, state):
        """ Epsilon-greedy action selection. """
        if self.random_state.rand() < self.epsilon:
            return self.random_state.choice(self.actions) # Explore
        else:
            # Exploit: Choose action with highest Q-value for the state
            q_values = [self.q_table[(state, a)] for a in self.actions]
            max_q = max(q_values)
            # Break ties randomly
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return self.random_state.choice(best_actions)

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        
        if not self.current_bids or not self.current_asks:
            return

        # Track mid price for state calculation
        best_bid = self.current_bids[0][0]
        best_ask = self.current_asks[0][0]
        mid_price = round((best_bid + best_ask) / 2)
        self.mid_price_history.append(mid_price)
        # Keep history small
        if len(self.mid_price_history) > 5:
            self.mid_price_history.pop(0)

        # Get current state and reward
        current_state = self.get_current_state()
        current_portfolio_value = self.markToMarket(self.holdings)
        
        # Q-Learning Update Rule
        if self.last_state is not None and self.last_action is not None:
            reward = current_portfolio_value - self.last_portfolio_value
            
            old_q = self.q_table[(self.last_state, self.last_action)]
            next_max_q = max([self.q_table[(current_state, a)] for a in self.actions])
            
            # Bellman Equation
            new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
            self.q_table[(self.last_state, self.last_action)] = new_q

        # Select and Execute Action
        action = self.choose_action(current_state)
        bid_price = self.current_bids[0][0] # should change if the agent decides to trade at a different price level
        ask_price = self.current_asks[0][0] # should change if the agent decides to trade at a different price level


        if(self.enforcer_enabled):
            # Passing the action through the Enforcer first
            portfolio_value = self.markToMarket(self.holdings)
            bid_liq, ask_liq = self.getKnownLiquidity(self.symbol)
            liquidity = bid_liq + ask_liq
            best_bid = self.current_bids[0][0]
            best_ask = self.current_asks[0][0]
            stock_price = round((best_bid + best_ask) / 2)

            new_action = self.enforcer.validate(action=action,
                                                bid_price=bid_price,
                                                ask_price=ask_price,
                                                portfolio_value=portfolio_value,
                                                liquidity=liquidity,
                                                stock_price=stock_price )

            if new_action != action:
                log_print(
                    "[Enforcer] Agent {} action OVERRIDDEN: {} -> {}",
                    self.name, self.actions_map[action], self.actions_map[new_action]
                )

            action = new_action 

        if action == 1: # BUY
            self.placeLimitOrder(self.order_size, True, bid_price)
        elif action == 2: # SELL
            self.placeLimitOrder(self.order_size, False, ask_price)
        elif action == 3: # CANCEL
            self.cancelAllOrders()
        # if action == 0 (HOLD), do nothing
        
        # Save state and value for next timestep
        self.last_state = current_state
        self.last_action = action
        self.last_portfolio_value = current_portfolio_value

    def kernelStopping(self):
        # Optional: Print Q-Table at end of simulation to see what it learned
        super().kernelStopping()
        log_print(f"\n[RLAgent] Final Q-Table for {self.name}:")
        for state_action, value in self.q_table.items():
            log_print(f"State: {state_action[0]}, Action: {state_action[1]}, Q-Value: {value:.2f}")
        if self.q_table_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.q_table_path)), exist_ok=True)
            with open(self.q_table_path, 'wb') as f:
                pickle.dump(self.q_table, f)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        
        if msg.body['msg'] == 'ORDER_EXECUTED':
            order = msg.body['order']
            if(self.enforcer_enabled):
                self.enforcer.validate( 
                    action=0,  # Action is not relevant for execution feedback
                    bid_price=-1    ,  # Price is not relevant for execution feedback
                    ask_price=-1    ,  # Price is not relevant for execution feedback
                    portfolio_value=-1,  # Portfolio value is not relevant for execution feedback
                    liquidity=-1    ,  # Liquidity is not relevant for execution feedback
                    stock_price=-1  ,  # Stock price is not relevant for execution feedback
                    trade_executed=True  # Inform the enforcer that a trade was executed
                )
            
            log_print(
                "*** TRADE EXECUTED *** Agent: {} | Side: {} | Qty: {} | Price: {}",
                self.name,
                "BUY" if order.is_buy_order else "SELL",
                order.quantity,
                order.fill_price
            )