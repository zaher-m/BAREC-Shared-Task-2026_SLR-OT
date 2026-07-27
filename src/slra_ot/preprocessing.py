"""The two text variants the members are trained on.

raw   : light normalisation only (kashida, whitespace). Diacritics are kept, they seem to
        carry signal at the low levels.
d3tok : CAMeL Tools morphological segmentation (base + clitics), same scheme the 2025
        top-ranked system used.
"""
import re

KASHIDA = "ـ"
_WS = re.compile(r"\s+")

VARIANTS = ("raw", "d3tok")


def light_norm(s: str) -> str:
    return _WS.sub(" ", str(s).replace(KASHIDA, "")).strip()


def build_d3tok():
    """Build the d3tok function. Loading the MLE disambiguator takes a while."""
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer
    from camel_tools.tokenizers.word import simple_word_tokenize

    mle = MLEDisambiguator.pretrained("calima-msa-r13")
    tok = MorphologicalTokenizer(disambiguator=mle, scheme="d3tok", split=True, diac=False)

    def fn(s: str) -> str:
        s = light_norm(s)
        if not s:
            return s
        return " ".join(tok.tokenize(simple_word_tokenize(s)))

    return fn


def transform_fn(variant):
    return build_d3tok() if variant == "d3tok" else light_norm
