"""Ordered stages of one sheet analysis: prepare -> trust -> propose -> explain -> assemble."""
from __future__ import annotations

from kalos.analysis.stages.assemble import assemble_result
from kalos.analysis.stages.explain import (
    DriversResult,
    PostFitDiagnostics,
    SurrogateExplanation,
    compute_drivers,
    compute_post_fit_diagnostics,
    explain_surrogate,
)
from kalos.analysis.stages.prepare import PreparedAnalysis, prepare_analysis
from kalos.analysis.stages.propose import ProposalResult, run_proposals
from kalos.analysis.stages.trust import TrustEvaluation, evaluate_trust

__all__ = [
    "DriversResult",
    "PostFitDiagnostics",
    "PreparedAnalysis",
    "ProposalResult",
    "SurrogateExplanation",
    "TrustEvaluation",
    "assemble_result",
    "compute_drivers",
    "compute_post_fit_diagnostics",
    "evaluate_trust",
    "explain_surrogate",
    "prepare_analysis",
    "run_proposals",
]
