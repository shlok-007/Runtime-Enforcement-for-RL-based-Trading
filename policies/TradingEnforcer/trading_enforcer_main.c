#include "F_TradingEnforcer.h"
#include <stdio.h>
#include <stdbool.h>

void print_state(int step, inputs_TradingEnforcer_t *in, outputs_TradingEnforcer_t *out, const char *prefix) {
    printf("%s Step %2d | Inputs [DD:%d, Illiq:%d, Lim:%d, Dev:%d] | Outputs [Trade:%d]\n", 
           prefix, step, 
           in->dd_exceeded, in->is_illiquid, in->will_exceed_limit, in->price_deviates,
           out->act_TRADE);
}

int main() {
    enforcervars_TradingEnforcer_t enf;
    inputs_TradingEnforcer_t inputs;
    outputs_TradingEnforcer_t outputs;

    // 1. Initialize variables
    TradingEnforcer_init_all_vars(&enf, &inputs, &outputs);

    printf("--- Starting TradingEnforcer Simulation ---\n");

    // 2. Run a simulation loop
    for (int i = 0; i < 15; i++) {
        // Reset inputs to safe defaults
        inputs.dd_exceeded = false;
        inputs.is_illiquid = false;
        inputs.will_exceed_limit = false;
        inputs.price_deviates = false;

        // Propose a generic BUY action every step
        outputs.act_TRADE = true;
        outputs.act_BUY = true;

        // 3. Inject Scenarios
        if (i == 3) {
            printf("\n[Scenario] Drawdown Exceeded triggered!\n");
            inputs.dd_exceeded = true;
        } else if (i == 6) {
            printf("\n[Scenario] Market is Illiquid!\n");
            inputs.is_illiquid = true;
        } else if (i == 9) {
            printf("\n[Scenario] Price Deviates significantly!\n");
            inputs.price_deviates = true;
        }

        print_state(i, &inputs, &outputs, "BEFORE");

        // 4. Run the Enforcer
        // This will call TradingEnforcer_run (our stub below) internally, 
        // then apply the safety policy.
        TradingEnforcer_run_via_enforcer(&enf, &inputs, &outputs);

        print_state(i, &inputs, &outputs, "AFTER ");
    }

    return 0;
}

// The enforcer library requires this function to be defined.
// In a real application, this contains the control logic (RL Agent).
// Here, we leave it empty because we set the 'proposed' outputs manually in main().
void TradingEnforcer_run(inputs_TradingEnforcer_t* inputs, outputs_TradingEnforcer_t* outputs) {
    // Pass-through or logic generation happens here normally.
}