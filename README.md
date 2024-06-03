# MIMO PID Tuning via Iterated LMI Restriction

This repository contains an implementation of the Multiple-Input Multiple-Output (MIMO) Proportional-Integral-Derivative (PID) controller tuning method described in the paper: ["MIMO PID tuning via iterated LMI restriction" by Stephen Boyd, Martin Hast, and Karl Johan Åström.](https://web.stanford.edu/~boyd/papers/pdf/mimo_pid_tuning.pdf)

## Usage

To use the MIMO PID tuning code:

1. Make sure you have the required dependencies installed (`numpy`, `control`, `cvxpy`).
2. Define your transfer function matrix `P` using the `control` library in `robust_mimo_main`.
3. Specify the desired frequency range `w` and the specifications for the sensitivity function `Smax`, complementary sensitivity function `Tmax`, and control effort `Qmax`.
4. The main function will call the `auto_mimo_pid` function with the appropriate arguments to tune the MIMO PID controller.
5. Set your desired number of datapoints `N` over the evaluating frequency range and the maximum number of iterations `max_iter`.
6. Main function will analyze the resulting closed-loop transfer functions and plot the desired frequency responses.


## Reference

<!-- This implementation is based on the following paper:

["MIMO PID tuning via iterated LMI restriction" by Stephen Boyd, Martin Hast, and Karl Johan Åström.](https://web.stanford.edu/~boyd/papers/pdf/mimo_pid_tuning.pdf) 
Please refer to the original paper for more details on the theoretical background and derivation of the method. -->
This repository also make use of the matlab implementation of  [MIMO PID Tune](https://github.com/rubindan/mimoPIDtune).
Please note that the current release of this repository may have some issues.