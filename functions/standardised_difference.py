import pandas as pd
import numpy as np

def standardised_difference_function (df: pd.DataFrame, sensitive_attribute: str, var_name: str, favoured_class: str or int, deprived_class: str or int):
    '''
    Measurement of the effect size between two groups. Particularly, the standardised difference
    for categorical variables with m > 2 classes as proposed by Dongsheng, Y. and Dalton, E. (2012)

    ------------------------------------------------
    Input:
    :df: pd.DataFrame                   -> dataframe with data relevant columns
    :sensitive_attribute: string        -> name of the binary flag indicating the treatment/control
    :var_name: string                   -> name of the categorical variable to calculate the SD
    :favoured_class: string or integer  -> label of the control obs contained in 'sensitive_attribute'
    :deprived_class: string or integer  -> label of the treatment obs contained in 'sensitive_attribute'

    ------------------------------------------------
    Return:
    :return: None       -> in case of a problem with the input (also raise a warning)
    :d: float           -> effect size
    :conf_interval:     -> confident interval of the estimation
    '''
    # If numerical variable, calculate the standardised mean difference
    if df[var_name].dtype in ['int64', 'float64']:

        t = df.loc[df[sensitive_attribute] == deprived_class, var_name]
        n_treatment = len(t)
        c = df.loc[df[sensitive_attribute] == favoured_class, var_name]
        n_control = len(c)
        d = (t.mean() - c.mean()) / np.sqrt(0.5 * (t.var() + c.var()))
        

    elif df[var_name].dtype == 'object': 
        
        # Calculate the conditional probabilities of each class in the protected attributes variable
            
        # For the favoured class
        control = df.loc[df[sensitive_attribute] == favoured_class, [sensitive_attribute, var_name]].rename(columns = {sensitive_attribute: 'control'})
        n_control = len(control)
        control = control.groupby(control[var_name]).count()/len(control)
        # For the deprived class
        treatment = df.loc[df[sensitive_attribute] == deprived_class, [sensitive_attribute, var_name]].rename(columns = {sensitive_attribute: 'treatment'})
        n_treatment = len(treatment)
        treatment = treatment.groupby(treatment[var_name]).count()/len(treatment)
        # Merge both results in a single table as suggested by  Dongsheng, Y. and Dalton, E (2012)
        conditional_probs = pd.merge(control, treatment, on = var_name)
        
        # If 2 classes in the variable, use two sample proportation formula
        if len(df[var_name].unique()) == 2:

            conditional_probs = conditional_probs.iloc[-1:]

            # Get treatment and control values
            c = conditional_probs['control'].values
            t = conditional_probs['treatment'].values

            d = (t - c)/(np.sqrt(t * (1 - t) + c * (1 - c))/2)
        
        # If >3 classes in the variable
        elif len(df[var_name].unique()) >= 3:
            # Get the number of K for the construction of the COV matrix. K = num of classes
            k = len(np.unique(df[var_name]))
            m = k - 1

            conditional_probs = conditional_probs.iloc[:m]
            # Clear indexes 
            temp = conditional_probs.reset_index(drop=True)
            
            # Calculate a (m − 1) × (m − 1) covariance matrix S

            s = []

            for i in range(m):
                a = []
                for j in range(m):
                    if i == j:
                        val =  0.5 * (temp['treatment'][i] * (1 - temp['treatment'][i]) + temp['control'][i] * (1 - temp['control'][i]))                  
                    else:
                        val = -0.5 * (temp['treatment'][i] * temp['treatment'][j] + temp['control'][i] * temp['control'][j])   
                    a.append(val)
                s.append(a)
            
            S = np.linalg.inv(np.array(s))
            S = np.matrix(S)   
            # Get treatment and control vectors
            c = np.matrix(conditional_probs['control'])
            t = np.matrix(conditional_probs['treatment'])
            # calculate the mahalanobis distance by defining the vectors for the control and treatment groups
            # Calculate the standardised difference 
            d = float(np.sqrt((t-c) * S * ((t-c).T)))
    else: 

        import warnings
        warnings.warn('Incorrect data type. Remember: "int64" or "float64" for continuous and "object" for categorical variables')
        return None
    
    # Calculation of sigma for a 95% confidence interval
    sigma = np.sqrt((n_treatment + n_control)/(n_treatment * n_control) + d**2/(2 * (n_treatment + n_control)))
    # upper and lower bounds of the CI
    upper = d + 1.96 * sigma
    lower = d - 1.96 * sigma

    conf_interval = (lower, upper)
    # Standard errors calculation as indicated by Alman, D. (2011) doi.org/10.1136/bmj.d2304 
    SE = (upper - lower)/(2 * 1.96)
    # calculate the z test statistic
    z = d/SE
    # calculate the p-value
    p_value = np.exp(-0.717 * z - 0.416 * z**2)

    temp_dict = dict(standardised_difference = d, 
                    CI = conf_interval, 
                    p_value = p_value)

    return temp_dict