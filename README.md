# MIMO PID Tuning via Iterated LMI Restriction

This repository contains an implementation of the Multiple-Input Multiple-Output (MIMO) Proportional-Integral-Derivative (PID) controller tuning method described in the paper: ["MIMO PID tuning via iterated LMI restriction" by Stephen Boyd, Martin Hast, and Karl Johan Åström.](https://web.stanford.edu/~boyd/papers/pdf/mimo_pid_tuning.pdf) 

## Main Code

The main code for tuning the MIMO PID controller is provided in the file `auto_mimo_pid.py`. It includes the following key components:

- Defining the transfer function matrix `P` using the `control` library.
- Specifying the frequency range `w` and the desired specifications `Smax`, `Tmax`, and `Qmax`.
- Calling the `auto_mimo_pid` function to tune the MIMO PID controller based on the specified plant, frequency range, and specifications.
- Analyzing the resulting closed-loop transfer functions `S`, `T`, and `Q`.
- Plotting the sensitivity function, complementary sensitivity function, and control effort.

## Usage

To use the MIMO PID tuning code:

1. Make sure you have the required dependencies installed (`numpy`, `matplotlib`, `control`).
2. Import the `auto_mimo_pid` function from the `auto_mimo_pid` module.
3. Define your transfer function matrix `P` using the `control` library.
4. Specify the desired frequency range `w` and the specifications `Smax`, `Tmax`, and `Qmax`.
5. Call the `auto_mimo_pid` function with the appropriate arguments to tune the MIMO PID controller.
6. Analyze the resulting closed-loop transfer functions and plot the desired frequency responses.


## Reference

This implementation is based on the following paper:

["MIMO PID tuning via iterated LMI restriction" by Stephen Boyd, Martin Hast, and Karl Johan Åström.](https://web.stanford.edu/~boyd/papers/pdf/mimo_pid_tuning.pdf) 
Please refer to the original paper for more details on the theoretical background and derivation of the method.

This repository also make use of the matlab implementation of  [MIMO PID Tune](https://github.com/rubindan/mimoPIDtune). However, please note that the current release of this repository may have some issues.