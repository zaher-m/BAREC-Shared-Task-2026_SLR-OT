"""Sentence-level Arabic readability assessment, BAREC 2026 Open Track.

  paths         where everything lives on disk
  metrics       BAREC metrics (QWK, Acc19/7/5/3, +-1, MAE)
  thresholds    cutting a score into levels, and the rules for fitting the cuts
  combiners     turning member scores into one score, plus cross-validation
  members       loading and aligning cached member scores
  doc_context   document-level corrections (exp001, negative)
  preprocessing raw and d3tok text variants
  folds         document-grouped folds over all the labelled data
  modeling      encoder member, and the six ordinal objectives
  modeling_llm  decoder-LLM member (LoRA + pooled head)
"""
__version__ = "1.0.0"
