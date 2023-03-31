import time
from memory_profiler import memory_usage
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

# written by Nicole Streltsov, Ritvik Jayanthi and Akriti Sharma March 2023

def get_memory_usage_and_runtime(train_function, arguments):
    '''
    gets peak memory usage of the training function in MB and runtime in seconds
    
    @param train_function: the model's training loop
    @param arguments: tuple of arguments for the function
    @return runtime (s), peak_mem (MB), result (return of the function)

    NOTE - if the function has no return values or arguments, run it like this: 
    mem = memory_usage(proc=train_function)
    '''

    s = time.time()
    mem, result = memory_usage((train_function, arguments), retval=True)
    e = time.time()
    runtime = e-s
    peak_mem = max(mem)
    
    return runtime, peak_mem, result


def get_epsilon_momentents_gaussian_dp(num_train_data, noise_multiplier, num_epochs, batch_size):
    '''
    calculates epsilon (ε) of the differentially private model. The model is said to be ε-differentially private if the model cannot differentiate 
    between two datasets that differ by one point. 

    @param num_training_data (int): the number of training examples
    @param noise_multiplier (double): standard deviation of the gaussian noise added to the gradients 
    @param num_epochs (int): number of epochs the model ran for
    @param batch_size (int): batch size used
    @returns epsilon(int) : epsilon value

    NOTE - to disable warning messages add this: 
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    '''
    delta = 1e-5 #typical value for delta 

    # compute the privacy budget
    epsilon, __ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
        n=num_train_data,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=num_epochs,
        delta=delta,
    )
    return epsilon

def dump_metrics_to_json(filename, runtime, peak_mem, test_accuracy, epsilon=0, is_dp=False):
    '''
    dumps all the metrics from the model to a text file with the name filename

    @param filename (string): filename, must end with .txt
    @param runtime (float): runtime of training in seconds
    @param peak_mem (float): peak memory of training in MB
    @param peak_mem (float): accuracy of model, in percentage form (eg. 91.23%)
    @param epsilon (float): epsilon from gaussian DP
    @param is_dp (boolean): true if the model had differential privacy incorporated
    '''

    if is_dp:
        privacy_utility = epsilon/test_accuracy
        metrics = f"Runtime(s): {runtime}\nPeak Mem(MB): {peak_mem}\nTest Accuracy: {test_accuracy}\nEpsilon: {epsilon}\nUtility: {privacy_utility}"
    else:
        metrics = f"Runtime(s): {runtime}\nPeak Mem(MB): {peak_mem}\nTest Accuracy: {test_accuracy}"

    with open(filename, 'w') as f:
        f.write(metrics)
