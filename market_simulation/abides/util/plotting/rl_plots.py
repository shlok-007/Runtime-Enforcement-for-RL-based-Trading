import pandas as pd
import matplotlib.pyplot as plt

def extract_portfolio_values(log_path, symbol='ABM'):
    """Extract portfolio value over time from an RL agent log."""
    df = pd.read_pickle(log_path, compression='bz2')
    
    # Filter for HOLDINGS_UPDATED events to track portfolio changes over time
    holdings_df = df[df['EventType'] == 'HOLDINGS_UPDATED'].copy()
    
    times = []
    values = []
    
    for event_time, row in holdings_df.iterrows():
        h = row['Event']
        cash = h.get('CASH', 0)
        times.append(event_time)  # EventTime is the INDEX, not a column
        values.append(cash / 100)

    return pd.Series(values, index=times)

# Alternatively, use MARKED_TO_MARKET events directly:
def extract_marked_to_market(log_path):
    """Extract mark-to-market portfolio value over time."""
    df = pd.read_pickle(log_path, compression='bz2')
    mtm = df[df['EventType'] == 'MARKED_TO_MARKET'].copy()
    values = mtm['Event'].astype(float) / 100  # cents -> dollars
    # values.index = mtm['EventTime']
    return values

# Load both logs
# no_enf = extract_marked_to_market('log/rmsc_trade_test_wo_enf/RL_AGENT.bz2')
# with_enf = extract_marked_to_market('log/rmsc_trade_test_w_enf/RL_AGENT.bz2')
# with_enf_complex = extract_marked_to_market('log/rmsc_trade_w_enf_complex/RL_AGENT.bz2')

no_enf = extract_marked_to_market('log/rmsc_trade_enf_pen/RL_AGENT_wo_ENF.bz2')
with_enf_wo_pen = extract_marked_to_market('log/rmsc_trade_enf_pen/RL_AGENT_w_ENF_wo_penalty.bz2')
with_enf_w_pen = extract_marked_to_market('log/rmsc_trade_enf_pen/RL_AGENT_w_ENF_5_percent_penalty.bz2')

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
no_enf.plot(ax=ax, label='Without Enforcer')
with_enf_wo_pen.plot(ax=ax, label='With Enforcer (No Penalty)')
with_enf_w_pen.plot(ax=ax, label='With Enforcer (5% Penalty)')
# with_enf_complex.plot(ax=ax, label='With Enforcer (Complex)')
ax.set_xlabel('Time')
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('RL Agent Portfolio Value: With vs Without Enforcer')
ax.legend()
plt.tight_layout()
plt.savefig('rl_portfolio_comparison.png', dpi=150)
plt.show()