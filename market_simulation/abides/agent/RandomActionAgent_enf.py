from agent.examples.ExampleExperimentalAgent import ExampleExperimentalAgentTemplate
from util.util import log_print
import pandas as pd
import numpy as np

from agent.enforcer.OTRWrapper import OTREnforcer

class RandomActionAgent(ExampleExperimentalAgentTemplate):
    def __init__(self, id, name, type, symbol, starting_cash, 
                 levels=5, subscription_freq=1e9, log_orders=False, random_state=None,
                 wake_freq='10s', order_size=100, 
                 prob_buy=0.33, prob_sell=0.33, prob_cancel=0.34):

        super().__init__(id, name, type, symbol, starting_cash, levels, subscription_freq, 
                         log_orders=log_orders, random_state=random_state)

        self.wake_freq = wake_freq
        self.order_size = order_size
        
        total = prob_buy + prob_sell + prob_cancel
        self.prob_buy = prob_buy / total
        self.prob_sell = prob_sell / total
        self.prob_cancel = prob_cancel / total

        self.enforcer = OTREnforcer(lib_path="./agent/libotr.so")
        self.has_executed_since_last_wake = False

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_freq)

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        
        if not self.current_bids or not self.current_asks:
            return


        # Enforcer
        # self.enforcer.update_input(self.has_executed_since_last_wake)

        action = self.random_state.choice(
            ['BUY', 'SELL', 'CANCEL'], 
            p=[self.prob_buy, self.prob_sell, self.prob_cancel]
        )

        allowed = self.enforcer.check_request(True, self.has_executed_since_last_wake)

        self.has_executed_since_last_wake = False

        if not allowed:
            log_print("Agent {} action BLOCKED by Enforcer (Tokens depleted)", self.name)

        if action == 'CANCEL':
            if len(self.orders) > 0:
                order_to_cancel = self.random_state.choice(list(self.orders.values()))
                self.cancelOrder(order_to_cancel)
                
        elif action == 'BUY':
            best_bid_price = self.current_bids[0][0]
            self.placeLimitOrder(self.order_size, True, best_bid_price)
            
        elif action == 'SELL':
            best_ask_price = self.current_asks[0][0]
            self.placeLimitOrder(self.order_size, False, best_ask_price*2)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        
        if msg.body['msg'] == 'ORDER_EXECUTED':
            order = msg.body['order']
            self.has_executed_since_last_wake = True
            log_print(
                "*** TRADE EXECUTED *** Agent: {} | Side: {} | Qty: {} | Price: {}",
                self.name,
                "BUY" if order.is_buy_order else "SELL",
                order.quantity,
                order.fill_price
            )