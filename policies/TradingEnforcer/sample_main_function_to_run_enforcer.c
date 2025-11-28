#include "F_prop1_and_prop2.h"
#include <stdio.h>

#include <time.h>
#define TICKS_PER_RUN 10

void print_data(uint32_t count, inputs_prop1_and_prop2_t inputs, outputs_prop1_and_prop2_t outputs) {
	printf("Tick %7d: vol: %d ; trade: %d\n", count, inputs.vol_high, outputs.act_TRADE );
}

int main() {
        clock_t start, end;

        start = clock();
        enforcervars_prop1_and_prop2_t enf;
        inputs_prop1_and_prop2_t inputs;
        outputs_prop1_and_prop2_t outputs;
        
        prop1_and_prop2_init_all_vars(&enf, &inputs, &outputs);

        uint32_t count = 0;

            inputs.vol_high = (count >= 7 && count <= 8);
            outputs.act_TRADE = count >= 5;
            printf("Before\n");
            print_data(count,inputs,outputs);
            prop1_and_prop2_run_via_enforcer(&enf, &inputs, &outputs);
            printf("After\n");
            print_data(count,inputs,outputs);
}

void prop1_and_prop2_run(inputs_prop1_and_prop2_t *inputs, outputs_prop1_and_prop2_t *outputs) {
    //do nothing
}

