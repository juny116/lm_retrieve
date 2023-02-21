import evaluate
from sklearn.metrics import ndcg_score
import datasets

_DESCRIPTION = """
Compute Normalized Discounted Cumulative Gain.
Sums the true scores ranked in the order induced by the predicted scores,
after applying a logarithmic discount. Then divides by the best possible
score (Ideal DCG, obtained for a perfect ranking) to obtain a score between
0 and 1.
This ranking metric returns a high value if true labels are ranked high by
``predictions``.
If a value for k is given to the metric, it will only consider the k highest
scores in the ranking 
References
    ----------
    
    `Wikipedia entry for Discounted Cumulative Gain
    <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_
    Jarvelin, K., & Kekalainen, J. (2002).
    Cumulated gain-based evaluation of IR techniques. ACM Transactions on
    Information Systems (TOIS), 20(4), 422-446.
    Wang, Y., Wang, L., Li, Y., He, D., Chen, W., & Liu, T. Y. (2013, May).
    A theoretical analysis of NDCG ranking measures. In Proceedings of the 26th
    Annual Conference on Learning Theory (COLT 2013).
    McSherry, F., & Najork, M. (2008, March). Computing information retrieval
    performance measures efficiently in the presence of tied scores. In
    European conference on information retrieval (pp. 414-421). Springer,
    Berlin, Heidelberg.
"""

_KWARGS_DESCRIPTION = """
Args:
    references ('list' of 'float'): True relevance 
    predictions ('list' of 'float'): Either predicted relevance, probability estimates or confidence values 
    k (int): If set to a value, only the k highest scores in the ranking will be considered, else considers all outputs.
        Defaults to None.
    sample_weight (`list` of `float`): Sample weights Defaults to None.
    ignore_ties ('boolean'): If set to true, assumes that there are no ties (this is likely if predictions are continuous)
        for efficiency gains. Defaults to False.
Returns:
    normalized_discounted_cumulative_gain ('float'): The averaged nDCG scores for all samples. 
        Minimum possible value is 0.0 Maximum possible value is 1.0
        
Examples:
    Example 1-A simple example
        >>> nDCG_metric = evaluate.load("ndcg")
        >>> results = nDCG_metric.compute(references=[[10, 0, 0, 1, 5]], predictions=[[.1, .2, .3, 4, 70]])
        >>> print(results)
        {'nDCG': 0.6956940443813076}
    Example 2-The same as Example 1, except with k set to 3.
        >>> nDCG_metric = evaluate.load("ndcg")
        >>> results = nDCG_metric.compute(references=[[10, 0, 0, 1, 5]], predictions=[[.1, .2, .3, 4, 70]], k=3)
        >>> print(results)
        {'nDCG@3': 0.4123818817534531}
    Example 3-There is only one relevant label, but there is a tie and the model can't decide which one is the one.
        >>> nDCG_metric = evaluate.load("ndcg")
        >>> results = nDCG_metric.compute(references=[[1, 0, 0, 0, 0]], predictions=[[1, 1, 0, 0, 0]], k=1)
        >>> print(results)
        {'nDCG@1': 0.5}
        >>> #That is it calculates both and returns the average of both
    Example 4-The Same as 3, except ignore_ties is set to True.
        >>> nDCG_metric = evaluate.load("ndcg")
        >>> results = nDCG_metric.compute(references=[[1, 0, 0, 0, 0]], predictions=[[1, 1, 0, 0, 0]], k=1, ignore_ties=True)
        >>> print(results)
        {'nDCG@1': 0.0}
        >>> # Alternative Result: {'nDCG@1': 1.0}
        >>> # That is it chooses one of the 2 candidates and calculates the score only for this one
        >>> # That means the score may vary depending on which one was chosen
"""

_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""

class nDCG(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value('float')),
                'references': datasets.Sequence(datasets.Value('float'))
            }),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html"],
        )

    def _compute(self, predictions, references, sample_weight=None, k=None, ignore_ties=False):
        results = {}
        if hasattr(k, "__iter__"):
            for i in k:
                score = ndcg_score(y_true=references,
                                y_score=predictions,
                                k=i,
                                sample_weight=sample_weight,
                                ignore_ties=ignore_ties
                                )
                results["nDCG@" + str(i)] = score
        else:
            score = ndcg_score(y_true=references,
                            y_score=predictions,
                            k=k,
                            sample_weight=sample_weight,
                            ignore_ties=ignore_ties
                            )
            if k is None:
                results["nDCG"] = score
            else:
                results["nDCG@" + str(k)] = score
        return results
