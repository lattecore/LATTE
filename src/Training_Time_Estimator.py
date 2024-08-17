class Training_Time_Estimator:
    def __init__(self):
        self.beta_key_fwd = {}
        self.beta_key_bwd = {}
        self.beta_non_fwd = 0
        self.beta_non_bwd = 0

    def load_profiled_betas(self, profiled_data):
        """
        Load beta value from profiled data.
        """
        self.beta_key_fwd = profiled_data['key_fwd']
        self.beta_key_bwd = profiled_data['key_bwd']
        self.beta_non_fwd = profiled_data['non_fwd']
        self.beta_non_bwd = profiled_data['non_bwd']

    def estimate_single_pass(self, algo_selection, C_key, C_non):
        """
        algo_selection: Result from Layer Algorithm Selector
        C_key: Computation Workload in Direct Algorithm of all key layers
        C_non: Total Computation Workload of Non-key layers
        """
        T_key = sum((self.beta_key_fwd[algo] + self.beta_key_bwd[algo]) * c
                    for algo, c in zip(algo_selection, C_key))
        T_non = (self.beta_non_fwd + self.beta_non_bwd) * C_non
        return T_key + T_non

    def estimate_training_time(self, algo_selection, C_key, C_non, num_epochs, batch_size, num_batches):
        """
        Estimate total training time
        """
        T = self.estimate_single_pass(algo_selection, C_key, C_non)
        return num_epochs * batch_size * num_batches * T

def create_estimator():
    return Training_Time_Estimator()

# Example
estimator = create_estimator()

# Profiled coefficient beta. You should profile by yourself, then replace all '0'.
profiled_data = {
    'key_fwd': {'algo1': 0, 'algo2': 0, 'algo3': 0, 'algo4': 0, 'algo5': 0, 'algo6': 0, 'algo7': 0, 'algo8': 0},
    'key_bwd': {'algo1': 0, 'algo2': 0, 'algo3': 0, 'algo4': 0, 'algo5': 0, 'algo6': 0, 'algo7': 0, 'algo8': 0},
    'non_fwd': 0,
    'non_bwd': 0
}
estimator.load_profiled_betas(profiled_data)

# Result from Layer_Algorithm_Selector
algo_selection = ['algo2', 'algo1', 'algo3', 'algo4', 'algo5', ...]

# Computation workload of key layers and non-key layers
C_key = [0, 0, 0, 0, 0, ...]  # You should test in your model, then replace all '0' and remove '...'
C_non = 0  # You should test in your model, then replace all '0'

# Estimate single-pass training time
T = estimator.estimate_single_pass(algo_selection, C_key, C_non)
print(f"Estimated single-pass latency: {T}")

# Estimate total training time, batch_size should also replaced by your setting
num_epochs = 10
batch_size = 32
num_batches = 100
T_train = estimator.estimate_training_time(algo_selection, C_key, C_non, num_epochs, batch_size, num_batches)
print(f"Estimated total training time: {T_train}")