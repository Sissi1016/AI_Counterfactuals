"""Checking counterfactual evaluators against quasi-random assignment of cases to decision makers.

- cfe.estimation: two-stage least squares with absorbed fixed effects and per-observation influence values, so that
  standard errors can be clustered by decision maker and combined across groups.
- cfe.check: the marginal-case check. A Design names the columns of a decision record (case, decision maker, decision,
  leniency, assignment cell, outcomes, applicant groups); PATENTS is the design used in this repository. Given any
  evaluator's scores, check() reports the evaluator's implied outcome for the marginal cases, its signed error, the
  same by applicant group, the balance of its predictions on leniency, and the group gap in its decisions.
"""
