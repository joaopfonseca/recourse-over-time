# Recourse Game

Implementation notes:
- [] The dependencies ``actionable-recourse`` and ``mip`` should be removed in
    the future. The package is no longer maintained.
- [] Test for both ``y_desired``=1 and ``y_desired``=0

Discussion topics:
- Understand how concept drift can be studied/applied here
- Regarding agents' adaptation:
    1. Compute Avg distance to counterfactuals;
        * Should Avg distance consider past time steps as well?
        * If so, how do we define the weights between the current time step
            and past time steps?
        * How should we calculate the distance when there are categorical
            (binary) features?
    2. Generate adaptation ratios;
        * Should there be a correlation between the adaptation ratio and
            proximity to the goal?
    3. For each agent, do ``adaptation * avg distance``
- Literature analysis
