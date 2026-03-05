import ctypes
import os

# 1. Define C Types to match F_OTR.h
class OTR_Policy_States(ctypes.c_int):
    # Mapping enum values
    s_normal = 0
    s_throttled = 1
    violation = 2

class Inputs_OTR(ctypes.Structure):
    _fields_ = [("act_EXEC", ctypes.c_bool)]

class Outputs_OTR(ctypes.Structure):
    _fields_ = [("act_MSG", ctypes.c_bool)]

class EnforcerVars_OTR(ctypes.Structure):
    _fields_ = [
        ("_policy_OTR_Policy_state", ctypes.c_int), # enum is int
        ("tokens", ctypes.c_int32)
    ]

class OTREnforcer:
    def __init__(self, lib_path="./libotr.so"):
        # 2. Load the Shared Library
        try:
            # Load absolute path to ensure it's found
            abs_path = os.path.abspath(lib_path)
            self.lib = ctypes.CDLL(abs_path)
        except OSError as e:
            print(f"Error loading enforcer library at {abs_path}: {e}")
            raise

        # 3. Define Argument Types for C Functions
        # void OTR_init_all_vars(enforcervars_OTR_t* me, inputs_OTR_t* inputs, outputs_OTR_t* outputs);
        self.lib.OTR_init_all_vars.argtypes = [
            ctypes.POINTER(EnforcerVars_OTR),
            ctypes.POINTER(Inputs_OTR),
            ctypes.POINTER(Outputs_OTR)
        ]

        # void OTR_run_input_enforcer_OTR_Policy(enforcervars_OTR_t* me, inputs_OTR_t* inputs);
        self.lib.OTR_run_input_enforcer_OTR_Policy.argtypes = [
            ctypes.POINTER(EnforcerVars_OTR),
            ctypes.POINTER(Inputs_OTR)
        ]

        # void OTR_run_output_enforcer_OTR_Policy(enforcervars_OTR_t* me, inputs_OTR_t* inputs, outputs_OTR_t* outputs);
        self.lib.OTR_run_output_enforcer_OTR_Policy.argtypes = [
            ctypes.POINTER(EnforcerVars_OTR),
            ctypes.POINTER(Inputs_OTR),
            ctypes.POINTER(Outputs_OTR)
        ]

        self.lib.OTR_run_via_enforcer.argtypes = [
            ctypes.POINTER(EnforcerVars_OTR),
            ctypes.POINTER(Inputs_OTR),
            ctypes.POINTER(Outputs_OTR)
        ]

        # 4. Initialize State
        self.enf_vars = EnforcerVars_OTR()
        self.inputs = Inputs_OTR()
        self.outputs = Outputs_OTR()
        
        # Initialize C memory
        self.lib.OTR_init_all_vars(
            ctypes.byref(self.enf_vars),
            ctypes.byref(self.inputs),
            ctypes.byref(self.outputs)
        )

    def update_input(self, execution_occurred: bool):
        """Run the Input Enforcer (Update Token Bucket based on Executions)"""
        self.inputs.act_EXEC = execution_occurred
        self.lib.OTR_run_input_enforcer_OTR_Policy(
            ctypes.byref(self.enf_vars),
            ctypes.byref(self.inputs)
        )

    def check_request(self, wants_to_message: bool, was_trade_executed: bool) -> bool:
        """Run the Output Enforcer (Check if Message is allowed)"""
        # Set the 'Proposed' output
        self.outputs.act_MSG = wants_to_message
        self.inputs.act_EXEC = was_trade_executed
        
        # Run Enforcer Logic
        # self.lib.OTR_run_output_enforcer_OTR_Policy(
        #     ctypes.byref(self.enf_vars),
        #     ctypes.byref(self.inputs),
        #     ctypes.byref(self.outputs)
        # )

        self.lib.OTR_run_via_enforcer(
            ctypes.byref(self.enf_vars),
            ctypes.byref(self.inputs),
            ctypes.byref(self.outputs)
        )
        
        # Return the 'Enforced' output
        return self.outputs.act_MSG

    def get_tokens(self):
        return self.enf_vars.tokens