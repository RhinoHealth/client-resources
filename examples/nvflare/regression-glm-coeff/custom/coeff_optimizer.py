import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


class CoeffOptimizer:

    def name(self):
        raise NotImplementedError

    @staticmethod
    def get_glm_obj(formula, offset, family_class, data=None, data_y=None, data_x=None) -> sm.GLM:
        offset = np.log(data[offset] + 1e-10) if offset else None
        if formula:
            glm = smf.glm(formula, data, family=family_class(), offset=offset)
        else:
            glm = sm.GLM(data_y, data_x, family=family_class(), offset=offset)
        return glm

    def get_local_coeffs(self, current_round, np_data, formula, offset, family_class, logger_warnings, data=None, data_y=None, data_x=None, site_info=None):
        raise NotImplementedError

    def add(self, data, add_results, contribution_round):
        raise NotImplementedError

    def get_result(self, **kwargs):
        raise NotImplementedError


class NewtonRaphson(CoeffOptimizer):

    def __init__(self, **kwargs):
        self.accuracy_threshold = None

    def name(self):
        return "NR"

    def get_local_coeffs(self, current_round, np_data, formula, offset, family_class, logger_warnings, data=None, data_y=None, data_x=None, site_info=None):
        """
        This function is called by the client to calculate the local coefficients, and modifies the np_data object with the results.
        """
        if current_round == 0:
            np_data["method"] = self.name()
            glm = self.get_glm_obj(formula, offset, family_class, data, data_y, data_x)
            res = glm.fit()
            beta = res.params
            np_data['beta'] = beta.values

        elif 'beta' in np_data:
            fed_beta = np_data['beta']
            glm = self.get_glm_obj(formula, offset, family_class, data, data_y, data_x)
            first_derivative = glm.score(params=fed_beta)
            second_derivative = glm.hessian(params=fed_beta)
            np_data["first_derivative"] = first_derivative
            np_data["second_derivative"] = second_derivative

    @staticmethod
    def get_add_results_base_dict():
        return {
            "betas_list": [],
            "beta_opt": 0,
            "first_derivative_sum": 0,
            "second_derivative_sum": 0,
            "count_clients": 0,

        }

    def add(self, data, add_results, contribution_round):
        if contribution_round == 0:
            add_results["beta_opt"] += data["beta"]
            add_results["count_clients"] += 1
        else:
            add_results["first_derivative_sum"] += data["first_derivative"]
            add_results["second_derivative_sum"] += data["second_derivative"]

    def get_accuracy_threshold(self, target_accuracy, prev_beta, **kwargs):
        return np.absolute(target_accuracy * prev_beta)

    def get_result(self, add_results, contribution_round, target_accuracy, **kwargs):
        accuracy_threshold = kwargs.get("accuracy_threshold")
        if contribution_round == 0:
            prev_beta = add_results["beta_opt"] / add_results["count_clients"]
            accuracy_threshold = self.get_accuracy_threshold(target_accuracy, prev_beta)
            fed_stderror = np.zeros(prev_beta.shape)
            next_beta = prev_beta
        else:
            prev_beta = add_results["betas_list"][-1]
            second_derivative_sum_inverse = np.linalg.inv(add_results["second_derivative_sum"])
            iteration_step = np.matmul(add_results["first_derivative_sum"], second_derivative_sum_inverse) * -1
            next_beta = prev_beta + iteration_step
            accuracy = np.absolute(next_beta - prev_beta)

            # calculate federated stderror
            fisher = -1 * add_results["second_derivative_sum"]
            fisher_inv = np.linalg.inv(fisher)
            fed_stderror = np.sqrt(np.diag(fisher_inv))

            # Stop if the result is already accurate enough
            if np.all(np.greater(accuracy_threshold, accuracy)):
                print(f"Reached accuracy threshold")
                return accuracy_threshold, {"beta": next_beta, "fed_stderror": fed_stderror, "signal": 'ABORT', "Reached accuracy threshold": True}

        add_results["first_derivative_sum"] = 0
        add_results["second_derivative_sum"] = 0
        add_results["betas_list"].append(next_beta)
        print(f"next beta after contribution round {contribution_round} is {next_beta}")
        return accuracy_threshold, {"beta": next_beta, "fed_stderror": fed_stderror, "Reached accuracy threshold": False}


class IRLS(CoeffOptimizer):

    def name(self):
        return "IRLS"

    @staticmethod
    def _site_irls_initialization(glm, start_params=None):
        """
        Initialize the model for the IRLS algorithm.
        This method is called once per site and runs locally at each site.
        """
        endog = glm.endog
        exog = glm.exog

        if start_params is None:
            mu = glm.family.starting_mu(glm.endog)
            lin_pred = glm.family.predict(mu)
        else:
            lin_pred = np.dot(exog, start_params) + glm._offset_exposure
            mu = glm.family.fitted(lin_pred)
        glm.scale = glm.estimate_scale(mu)
        dev = glm.family.deviance(endog, mu, glm.var_weights,
                                  glm.freq_weights, glm.scale)
        if np.isnan(dev):
            raise ValueError("The first guess on the deviance function "
                             "returned a nan.  This could be a boundary "
                             " problem and should be reported.")
        return glm, lin_pred, mu

    @staticmethod
    def _site_irls_iteration(site_info, logger_warnings):
        """
        Perform one iteration of the IRLS algorithm.
        This method is called during each iteration of IRLS and runs locally at each site.
        """
        glm = site_info['glm']

        if (params := site_info.get('params', None)) is None:
            # First iteration - take the initial lin_pred and mu
            lin_pred = site_info['initial_lin_pred']
            mu = site_info['initial_mu']
        else:
            # Advanced iteration - use the params to calculate lin_pred and mu
            wexog = site_info['wexog']
            wendog = site_info['wendog']
            wlsendog = site_info['wlsendog']

            fitted_values = glm.exog.dot(params)
            resid = wlsendog - fitted_values
            wresid = wendog - wexog.dot(params)
            df_resid = wexog.shape[0] - wexog.shape[1]
            scale = np.dot(wresid, wresid) / df_resid

            lin_pred = np.dot(glm.exog, params)
            lin_pred += glm._offset_exposure
            mu = glm.family.fitted(lin_pred)
            glm.scale = glm.estimate_scale(mu)
            if glm.endog.squeeze().ndim == 1 and np.allclose(mu - glm.endog, 0):
                msg = ("Perfect separation or prediction detected, "
                       "parameter may not be identified")
                logger_warnings(msg)

        # Recalculate weights and latest endog/exog for WLS
        glm.weights = (glm.iweights * glm.n_trials *
                       glm.family.weights(mu))
        wlsendog = (lin_pred + glm.family.link.deriv(mu) * (glm.endog - mu)
                    - glm._offset_exposure)

        w_half = np.sqrt(glm.weights)
        wendog = w_half * wlsendog
        if np.isscalar(glm.weights):
            wexog = w_half * glm.exog
        else:
            wexog = np.asarray(w_half)[:, None] * glm.exog

        # Store info for the next iteration (to be used locally)
        site_info["glm"] = glm
        site_info["wexog"] = wexog
        site_info["wendog"] = wendog
        site_info["wlsendog"] = wlsendog

        # Prepare the partial matrices to solve federated WLS (to be shared with the cloud)
        A = wexog.transpose().dot(wexog)
        B = wexog.transpose().dot(wendog)

        ols_params = {
            'A': A,
            'B': B,
        }

        return ols_params

    def get_local_coeffs(self, current_round, np_data, formula, offset, family_class, logger_warnings, data=None, data_y=None, data_x=None, site_info=None):
        """
        Calculate the local coefficients, and modifies the np_data object with the results.
        This method is called during each iteration of IRLS and runs locally at each site.
        """
        glm = self.get_glm_obj(formula, offset, family_class, data, data_y, data_x)
        if current_round == 0:
            np_data["method"] = self.name()
            glm, lin_pred, mu = self._site_irls_initialization(glm)
            site_info["glm"] = glm
            site_info["initial_lin_pred"] = lin_pred
            site_info["initial_mu"] = mu
            np_data["initial_beta"] = np.zeros(glm.exog.shape[1])
            np_data["exog_names"] = glm.exog_names
        else:
            site_info["params"] = np_data["site_info"]["params"]
        np_data["site_hessian"] = glm.hessian(params=np_data.get("site_info", {}).get("params", np_data.get("initial_beta")))
        np_data["site_ols_params"] = self._site_irls_iteration(site_info, logger_warnings)

    @staticmethod
    def get_add_results_base_dict():
        return {
            "beta_opt": None,
            "A_sum": 0,
            "B_sum": 0,
            "combined_hessian": 0,
        }

    def add(self, data, add_results, contribution_round):
        if add_results["beta_opt"] is None:
            add_results["beta_opt"] = data.get("initial_beta")  # Should be sent in the first round
        add_results["A_sum"] += data["site_ols_params"]['A']
        add_results["B_sum"] += data["site_ols_params"]['B']
        add_results["combined_hessian"] += data["site_hessian"]
        add_results["exog_names"] = data["exog_names"]

    def get_result(self, add_results, contribution_round, target_accuracy, **kwargs):
        next_beta = np.linalg.inv(add_results["A_sum"]).dot(add_results["B_sum"])
        # Prepare for next iteration
        accuracy = np.absolute(next_beta - add_results["beta_opt"])
        add_results["beta_opt"] = next_beta
        fisher_info = -1 * add_results["combined_hessian"]
        fisher_info_inv = np.linalg.inv(fisher_info)
        fed_stderror = np.sqrt(np.diag(fisher_info_inv))
        # Stop if the result is already accurate enough
        if np.all(np.greater(target_accuracy, accuracy)):
            print(f"Reached accuracy threshold")
            return None, {"beta": next_beta, "fed_stderror": fed_stderror, "variable_names_by_betas_order": data["exog_names"], "signal": 'ABORT', "Reached accuracy threshold": True}

        add_results["A_sum"] = 0
        add_results["B_sum"] = 0
        add_results["combined_hessian"] = 0
        print(f"next beta after contribution round {contribution_round} is {next_beta}")
        return None, {"site_info": {"params": next_beta}, "beta": next_beta, "fed_stderror": fed_stderror, "variable_names_by_betas_order": data["exog_names"], "Reached accuracy threshold": False}


OPTIMIZERS = {"NR": NewtonRaphson, "IRLS": IRLS}
