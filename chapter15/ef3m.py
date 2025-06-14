import math
import random


class EF3M:
    """
    EF3M: Exact Fit of the first 3 Moments
    This class implements the EF3M algorithm for fitting gaussian mixtures having 2 components.
    """

    def __init__(
        self,
        tol: float = 1e-5,
        factor: float = 5.0,
        convergence_type: str = "4th",
    ):
        """
        Initialize the EF3M class with the given moments.

        :param moments: A list of five moments [mean, variance, skewness(not normalized), kurtosis(not normalized), fifth_moment].
        :param tol: Tolerance for convergence.
        :param factor: Factor to adjust the search space. (lambda of the paper)
        :param convergence_type: Type of convergence to use. Either '4th' or '5th'.
        """
        if convergence_type not in ["4th", "5th"]:
            raise ValueError("convergence_type must be either '4th' or '5th'")
        self.convergence_type = convergence_type
        self.tol = tol
        self.factor = factor
        self.params = [0] * 5

    def fit(self, moments):
        """
        fit the parameters of the distribution based on the given moments.
        :param moments: A list of five moments [mean, variance, skewness(not normalized), kurtosis(not normalized), fifth_moment].
        """
        centered_std = self._get_centerd_moments(moments, 2) ** 0.5
        mu_2_range = [
            i * self.tol * self.factor * centered_std + moments[0]
            for i in range(1, int(1 / self.tol))
        ]
        error = sum([moments[i] ** 2 for i in range(5)])
        params = [0] * 5
        for mu_2 in mu_2_range:
            cur_params, cur_error = self._fit(mu_2, moments)
            if cur_error < error:
                error = cur_error
                params = cur_params
        return params

    def _fit(self, mu_2, moments):
        """
        fit the parameters of the distribution based on the given moments.
        :param mu_2: The mean of the second distribution.
        :param moments: A list of five moments [mean, variance, skewness(not normalized), kurtosis(not normalized), fifth_moment].
        """
        p_1 = random.uniform(0, 1)
        cur_iter = 0
        error = sum([moments[i] ** 2 for i in range(5)])
        params = [0] * 5
        while True:
            # Estimate the parameters of the Gaussian distributions
            gaussian_params = self._estimate_gaussian_parameters(mu_2, p_1, moments)
            cur_moments = self.get_moments(
                gaussian_params[0],
                gaussian_params[2],
                gaussian_params[1],
                gaussian_params[3],
                gaussian_params[4],
            )
            cur_error = sum([(moments[i] - cur_moments[i]) ** 2 for i in range(5)])
            if cur_error < error:
                error = cur_error
                params = gaussian_params
            if abs(p_1 - gaussian_params[4]) < self.tol:
                break
            if cur_iter > 1 / self.tol:
                break
            mu_2 = gaussian_params[2]
            p_1 = gaussian_params[4]
            cur_iter += 1
        return params, error

    def get_moments(self, mu_1, mu_2, sigma_1, sigma_2, p_1):
        """
        Calculate the moments of the Gaussian mixture.
        :param mu_1: The mean of the first distribution.
        :param mu_2: The mean of the second distribution.
        :param sigma_1: The standard deviation of the first distribution.
        :param sigma_2: The standard deviation of the second distribution.
        :param p_1: The mixing proportion of the first distribution.
        :return: A list of moments [mean, variance, skewness, kurtosis, fifth_moment].
        """
        m1 = p_1 * mu_1 + (1 - p_1) * mu_2
        m2 = p_1 * (sigma_1**2 + mu_1**2) + (1 - p_1) * (sigma_2**2 + mu_2**2)
        m3 = p_1 * (3 * sigma_1**2 * mu_1 + mu_1**3) + (1 - p_1) * (
            3 * sigma_2**2 * mu_2 + mu_2**3
        )
        m4 = p_1 * (3 * sigma_1**4 + 6 * sigma_1**2 * mu_1**2 + mu_1**4) + (
            1 - p_1
        ) * (3 * sigma_2**4 + 6 * sigma_2**2 * mu_2**2 + mu_2**4)
        m5 = p_1 * (
            15 * sigma_1**4 * mu_1 + 10 * sigma_1**2 * mu_1**3 + 3 * mu_1**5
        ) + (1 - p_1) * (
            15 * sigma_2**4 * mu_2 + 10 * sigma_2**2 * mu_2**3 + 3 * mu_2**5
        )
        return [m1, m2, m3, m4, m5]

    def _estimate_gaussian_parameters(self, mu_2, p_1, moments):
        """
        Estimate the parameters of the Gaussian distributions based on the given moments and p_1.
        :param mu_2: The mean of the second distribution.
        :param p_1: The mixing proportion of the first distribution.
        :return: A list of parameters [mu_1, sigma_1, mu_2, sigma_2, p_1].
        """
        m1, m2, m3, m4, m5 = moments

        mu_1 = (m1 - (1 - p_1) * mu_2) / p_1
        sigma_2 = (
            (
                m3
                + 2 * p_1 * (mu_1**3)
                + (p_1 - 1) * (mu_2**3)
                - 3 * mu_1 * (m2 + (mu_2**2) * (p_1 - 1))
            )
            / (3 * (1 - p_1) * (mu_2 - mu_1))
        ) ** 0.5
        sigma_1 = (m2 - sigma_2**2 - mu_2**2) / p_1 + sigma_2**2 + mu_2**2 - mu_1**2
        if self.convergence_type == "4th":
            p_1 = (m4 - 3 * sigma_2**4 - 6 * (sigma_2) ** 2 * (mu_2**2) - mu_2**4) / (
                3 * (sigma_1**4 - sigma_2**4)
                + 6 * (sigma_1**2 * mu_1**2 - sigma_2**2 * mu_2**2)
                + mu_1**4
                - mu_2**4
            )
        elif self.convergence_type == "5th":
            a = (
                6 * sigma_2**4
                + (m4 - p_1 * (3 * sigma_1**4 + 6 * sigma_1**2 * mu_1**2 + mu_1**4))
                / (1 - p_1)
            ) ** 0.5
            mu_2 = (-3 * sigma_2**4 + a) ** 0.5
            a = 15 * sigma_1**4 * mu_1 + 10 * sigma_1**2 * mu_1**3 + 3 * mu_1**5
            b = 15 * sigma_2**4 * mu_2 + 10 * sigma_2**2 * mu_2**3 + 3 * mu_2**5
            p_1 = (m5 - b) / (a - b)
        return mu_1, sigma_1, mu_2, sigma_2, p_1

    def _get_centerd_moments(self, moments, order):
        """
        Calculate the centered moments of the Gaussian mixture.
        :param moments: A list of moments [mean, variance, skewness, kurtosis, fifth_moment].
        :param order: The order of the moment to calculate.
        :return: The centered moment of the given order.
        """
        moment_c = 0
        for i in range(order + 1):
            moment_c += (
                (-1) ** i
                * math.comb(order, i)
                * moments[0] ** i
                * moments[order - i - 1]
            )
        return moment_c
