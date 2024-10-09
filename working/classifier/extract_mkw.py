"""Rule-based extraction of keywords that show modality of an image.

Taxonomy of keywords is given in [1].

References
----------
[1] Garcia Seco de Herrera, A., Muller, H. & Bromuri, S.
    "Overview of the ImageCLEF 2015 medical classification task."
    In Working Notes of CLEF 2015 (Cross Language Evaluation Forum) (2015).
"""

def load_taxonomy():
    """Load taxonomy of keywords."""
    pass


def extract_kw():
    """Find keywords in captions and assign probabilities.

    Based on the variety and number of keywords found in the caption,
    a probability is assigned to the caption describing an image of each given modality in the taxonomy.
    """
    pass


def eval_kw():
    """Evaluate assigned modalities to each image.

    Evaluation is done by random observation and human evaluation.
    """
    pass




if __name__ == "__main__"