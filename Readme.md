# Runtime Enforcement for RL-based Trading

## Easy-RTE Compile Instructions
First install Easy-RTE as per the instructions [here](https://github.com/PRETgroup/easy-rte#build-instructions).
```bash
cd policies/TradingEnforcer
easy-rte-parser -product -i TradingEnforcer.erte -o TradingEnforcer.xml
easy-rte-c -i TradingEnforcer.xml -o .
gcc -shared -o lib_TradingEnforcer.so -fPIC F_TradingEnforcer.c enforcer_impl.c
```

## Enforceability check for policies using NuSMV
Install NuSMV from [here](https://nusmv.fbk.eu/downloads.html).
```bash
cd nusmv
./NuSMV allPolicies.smv
```

## Instructions for running the simulator

```bash
cd market_simulation/abides
pip install -r requirements.txt
python -u abides.py -c rmsc03_w_rl -t ABM -d 20200603 -s 5787 --end-time 15:00:00 -l rmsc_trade_test --enf
```