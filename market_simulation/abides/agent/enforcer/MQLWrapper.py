import ctypes
import os

# 1. Define C Types to match F_MQL.h

# dtimer_t is uint64_t
dtimer_t = ctypes.c_uint64

class MQL_Policy_States(ctypes.c_int):
    # Mapping enum values
    s_empty = 0
    s_resting = 1
    s_violation = 2

class Inputs_MQL(ctypes.Structure):
    _fields_ = []  # No inputs defined in F_MQL.h

class Outputs_MQL(ctypes.Structure):
    _fields_ = [
        ("act_PLACE", ctypes.c_bool),
        ("act_CANCEL", ctypes.c_bool),
    ]

class EnforcerVars_MQL(ctypes.Structure):
    _fields_ = [
        ("_policy_MQL_Policy_state", ctypes.c_int),  # enum is int
        ("c_age", ctypes.c_uint64),                  # dtimer_t is uint64_t
    ]

class MQLEnforcer:
    def __init__(self, lib_path="./libmql.so"):
        # 2. Load the Shared Library
        try:
            abs_path = os.path.abspath(lib_path)
            self.lib = ctypes.CDLL(abs_path)
        except OSError as e:
            print(f"Error loading enforcer library at {abs_path}: {e}")
            raise

        # 3. Define Argument Types for C Functions
        # void MQL_init_all_vars(enforcervars_MQL_t* me, inputs_MQL_t* inputs, outputs_MQL_t* outputs);
        self.lib.MQL_init_all_vars.argtypes = [
            ctypes.POINTER(EnforcerVars_MQL),
            ctypes.POINTER(Inputs_MQL),
            ctypes.POINTER(Outputs_MQL)
        ]

        # void MQL_run_input_enforcer_MQL_Policy(enforcervars_MQL_t* me, inputs_MQL_t* inputs);
        self.lib.MQL_run_input_enforcer_MQL_Policy.argtypes = [
            ctypes.POINTER(EnforcerVars_MQL),
            ctypes.POINTER(Inputs_MQL)
        ]

        # void MQL_run_output_enforcer_MQL_Policy(enforcervars_MQL_t* me, inputs_MQL_t* inputs, outputs_MQL_t* outputs);
        self.lib.MQL_run_output_enforcer_MQL_Policy.argtypes = [
            ctypes.POINTER(EnforcerVars_MQL),
            ctypes.POINTER(Inputs_MQL),
            ctypes.POINTER(Outputs_MQL)
        ]

        # void MQL_run_via_enforcer(enforcervars_MQL_t* me, inputs_MQL_t* inputs, outputs_MQL_t* outputs);
        self.lib.MQL_run_via_enforcer.argtypes = [
            ctypes.POINTER(EnforcerVars_MQL),
            ctypes.POINTER(Inputs_MQL),
            ctypes.POINTER(Outputs_MQL)
        ]

        # 4. Initialize State
        self.enf_vars = EnforcerVars_MQL()
        self.inputs = Inputs_MQL()
        self.outputs = Outputs_MQL()

        # Initialize C memory
        self.lib.MQL_init_all_vars(
            ctypes.byref(self.enf_vars),
            ctypes.byref(self.inputs),
            ctypes.byref(self.outputs)
        )

    def check_request(self, wants_to_place: bool, wants_to_cancel: bool) -> tuple[bool, bool]:
        """Run the Output Enforcer (Check if PLACE/CANCEL actions are allowed)"""
        # Set the proposed outputs
        self.outputs.act_PLACE = wants_to_place
        self.outputs.act_CANCEL = wants_to_cancel

        # Run Enforcer Logic
        self.lib.MQL_run_via_enforcer(
            ctypes.byref(self.enf_vars),
            ctypes.byref(self.inputs),
            ctypes.byref(self.outputs)
        )

        # Return the enforced outputs
        return self.outputs.act_PLACE, self.outputs.act_CANCEL

    def get_age(self) -> int:
        """Return the current age (in ticks) of the resting order."""
        return self.enf_vars.c_age

    def get_state(self) -> str:
        """Return the current enforcer state as a string."""
        state_map = {
            MQL_Policy_States.s_empty: "empty",
            MQL_Policy_States.s_resting: "resting",
            MQL_Policy_States.s_violation: "violation",
        }
        return state_map.get(self.enf_vars._policy_MQL_Policy_state, "unknown")