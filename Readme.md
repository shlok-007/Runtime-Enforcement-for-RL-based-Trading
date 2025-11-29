## Compile Instructions

```bash
easy-rte-parser -product -i TradingEnforcer.erte -o TradingEnforcer.xml
easy-rte-c -i TradingEnforcer.xml -o .
gcc -shared -o libtradingenforcer.so -fPIC F_TradingEnforcer.c enforcer_impl.c
```