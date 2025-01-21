import numpy as np
from scipy.optimize import root
from scipy.constants import e, k


class Backend:
    def __init__(self):
        self.uploaded_data = None
        self.uploaded_file_name = None

    def load_data(self, file_name):
        """
        Load data from a file.

        Args:
            file_name (str): Path to the data file.

        Returns:
            bool: True if the file was successfully loaded, False otherwise.
        """
        try:
            # Load data from file
            data = np.loadtxt(file_name, delimiter=None, skiprows=1)  # Delimiter=None auto-detects
            if data.shape[1] < 2:
                raise ValueError("File must contain at least two columns: voltage and current.")
            self.uploaded_data = data
            self.uploaded_file_name = file_name.split("/")[-1]  # Extract file name
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def solve_shockley(self, V_values, I_sc, I_o, n, T, R_s, R_sh):
        """
        Solve the Shockley equation for a range of voltages.

        Args:
            V_values (np.ndarray): Array of voltage values.
            I_sc (float): Short-circuit current.
            I_o (float): Reverse saturation current.
            n (float): Ideality factor.
            T (float): Temperature in Kelvin.
            R_s (float): Series resistance.
            R_sh (float): Shunt resistance.

        Returns:
            np.ndarray: Array of current values corresponding to the input voltages.
        """
        # Initialize array for current values
        I_values = np.zeros_like(V_values)

        # Solve the Shockley equation for each voltage
        for i, V in enumerate(V_values):
            # Define the equation to solve
            def equation(I):
                return I_sc - I - I_o * (np.exp(e * (V + I * R_s) / (n * k * T)) - 1) - (V + I * R_s) / R_sh

            # Solve the equation using a numerical solver
            sol = root(equation, I_sc, method='lm')
            if sol.success:
                I_values[i] = -sol.x[0]  # Store the solution
            else:
                I_values[i] = np.nan  # Mark as invalid if the solver fails

        return I_values

    def calculate_r_squared(self, observed, predicted):
        """
        Calculate the R² value.

        Args:
            observed (np.ndarray): Observed current values.
            predicted (np.ndarray): Predicted current values.

        Returns:
            float: R² value.
        """
        # Calculate the mean of the observed values
        mean_observed = np.mean(observed)

        # Calculate total sum of squares (SS_total) and residual sum of squares (SS_residual)
        ss_total = np.sum((observed - mean_observed) ** 2)
        ss_residual = np.sum((observed - predicted) ** 2)

        # Handle division by zero
        if ss_total == 0:
            return 0.0

        # Calculate R²
        r_squared = 1 - (ss_residual / ss_total)
        return max(0.0, r_squared)  # Clip negative R² to zero

    def compute_log_data(self):
        """
        Compute the log of the absolute values of the uploaded data.

        Returns:
            np.ndarray: Log-transformed data (voltage and log(|current|)).
        """
        if self.uploaded_data is None:
            raise ValueError("No data uploaded.")

        # Compute log of absolute current values
        log_current = np.log10(np.abs(self.uploaded_data[:, 1]))

        # Combine voltage and log(|current|) into a new array
        log_data = np.column_stack((self.uploaded_data[:, 0], log_current))

        return log_data

    def compute_log_theoretical(self, V_values, I_values):
        """
        Compute the log of the absolute values of the theoretical current.

        Args:
            V_values (np.ndarray): Array of voltage values.
            I_values (np.ndarray): Array of theoretical current values.

        Returns:
            np.ndarray: Log-transformed theoretical data (voltage and log(|current|)).
        """
        if V_values.size == 0 or I_values.size == 0:
            raise ValueError("No theoretical data available.")

        # Compute log of absolute theoretical current values
        log_current = np.log10(np.abs(I_values))

        # Combine voltage and log(|current|) into a new array
        log_data = np.column_stack((V_values, log_current))

        return log_data
