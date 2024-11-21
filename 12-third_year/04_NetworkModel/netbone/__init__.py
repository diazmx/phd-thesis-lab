
"""
NetBone - Easy Backbone extraction library
==================

A Library that simplifies Extracting backbones from networks.
and graphs.
"""

from netbone.statistical.disparity import disparity
from netbone.structural.h_backbone import h_backbone
from netbone.statistical.noise_corrected import noise_corrected
from netbone.structural.doubly_stochastic import doubly_stochastic
from netbone.structural.high_salience_skeleton import high_salience_skeleton
from netbone.statistical.marginal_likelihood import MLF
from netbone.statistical.lans import lans
from netbone.structural.ultrametric_distance_backbone import ultrametric_distance_backbone
from netbone.structural.metric_distance_backbone import metric_distance_backbone
from netbone.structural.global_threshold import global_threshold
from netbone.structural.modulairy_backbone import modularity_backbone
from netbone.structural.maximum_spanning_tree import maximum_spanning_tree
from netbone.hybrid.glanb import glanb
from netbone.structural.pmfg import pmfg
from netbone.structural.plam import plam
from netbone.structural.mlam import mlam
from netbone.structural.gspar import gspar
from netbone.structural.degree import degree
from netbone.structural.betweenness import betweenness
# from netbone.structural.mad import mad
# # from netbone.statistical.correlation_and_statistic import correlation_and_statistic

from netbone.filters import threshold_filter, fraction_filter
from netbone import compare
from netbone import filters
from netbone import visualize
from netbone.backbone import Backbone
try:
    from netbone.statistical.maxent_graph.ecm_main import ecm
except ImportError:
    print("Can't load ECM Model in windows, try using it on linux")


def marginal_likelihood(data):
    data = data.copy()
    mlf = MLF(directed=False)
    return Backbone(mlf.fit_transform(data), method_name="Marginal Likelihood Filter", property_name="p_value", ascending=True, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')




# logger = logging.getLogger()
# logger.setLevel('DEBUG')



