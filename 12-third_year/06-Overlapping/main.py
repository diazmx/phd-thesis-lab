#from policy_mining import PolicyMining, introducir_ruido, crear_version_sparse
from policy_mining import PolicyMining
import sys

NAME_DATASET = sys.argv[1]
FILE_NAME = None
USER_ATTRS = None
RESOURCE_ATTRS = None
th_rule_sim = None

#f = open("output.txt", "w")
import policy_mining
print(policy_mining.PolicyMining.experiment_bridge_node_reassignment)


# Settings for each dataset
if NAME_DATASET == "AMZ":
    FILE_NAME = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    FILE_NAME = "/home/daniel/Documents/phd/phd-thesis-lab/12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    USER_ATTRS = ["MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME",
                  "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"]
    RESOURCE_ATTRS = ["RID"]
    th_rule_sim = 1

elif NAME_DATASET == "HC":
    FILE_NAME = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/02-HC/01-DistributionsCSV/HC-MOD.csv"
    USER_ATTRS = ['role', 'specialty', 'team', 'uward', 'agentfor']
    RESOURCE_ATTRS = ['type', 'patient', 'treatingteam',
                      'oward', 'author', 'topic']
    th_rule_sim = 1

elif NAME_DATASET == "CAV":
    FILE_NAME = "../00-Data/cav_policies.csv"
    USER_ATTRS = ["control", "monitoring", "fallback", "weather", "visibility",
                  "traffic_congestion"]
    RESOURCE_ATTRS = ["driving_task_loa", "vehicle_loa", "region_loa"]
    th_rule_sim = 1

elif NAME_DATASET == "IoT":
    FILE_NAME = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/11-IoT/01-DistributionsCSV/IoT-Rw.csv"
    USER_ATTRS = ["role", "age", "health"]
    RESOURCE_ATTRS = ["type", "area", "mode", "temperature", "lockstatus"]
    th_rule_sim = 1
    
else:
    print("Invalid Dataset")
    sys.exit()

th_big_com = 0.5
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
print(pm.experiment_bridge_node_reassignment.__code__.co_varnames)

pm.data_preprocessing(True, False)
pm.network_model()

# 👈 SE NECESITA ANTES DEL EXPERIMENTO
pm.community_detection(th_big_com, th_med_com)

# 👈 También se necesita antes
pm.rule_inference(th_rule_sim)

# 👇 AHORA SÍ puedes ejecutar el experimento
pm.experiment_bridge_node_reassignment(ratio_bridge_nodes=0.20,
                                        iterations=30,
                                        stability_threshold=0.40)
pm.evaluation()
res1 = pm.policy_refinement(th_rule_sim)
print(res1)

"""

th_big_com = 0.5
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection_nx(k_clique_value=3, big_threshold_ratio=0.6, med_threshold_ratio=0.3)
#pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
#pm.rule_inference_random(th_rule_sim)
pm.evaluation()
#pm.evaluation_random()
#f.write("Normal\n")
#f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res1 = pm.policy_refinement(th_rule_sim)
#res = pm.policy_refinement_random(th_rule_sim)
print(res1)


th_big_com = 0.5
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.5
th_med_com = 0.25
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.25
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.25
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.15
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.35
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.15
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.35
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

############ NOISY ############
f.write(" ======= NOISY =======\n")
th_big_com = 0.5
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.5
th_med_com = 0.25
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.25
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.25
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.15
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.35
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.15
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.35
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(True, False)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

#################### SPARSE
f.write(" ======= SPARSEE =======\n")
th_big_com = 0.5
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.5
th_med_com = 0.25
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.25
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.25
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.25
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.15
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.4
th_med_com = 0.35
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.15
th_rule_sim = 1
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

th_big_com = 0.6
th_med_com = 0.35
th_rule_sim = 2
pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, True)
pm.network_model()
pm.community_detection(th_big_com, th_med_com)
pm.community_detection_random(th_big_com, th_med_com)
pm.rule_inference(th_rule_sim)
pm.rule_inference_random(th_rule_sim)
pm.evaluation()
pm.evaluation_random()
f.write("Normal\n")
f.write("th_big_com="+str(th_big_com) + "-th_med_com="+str(th_med_com)+"th_rule_sim="+str(th_rule_sim)+"\n")
res = pm.policy_refinement(th_rule_sim)
f.write("Real"+str(res)+"\n")
res = pm.policy_refinement_random(th_rule_sim)
f.write("Random"+str(res))

f.write("\n\n")

f.close()

"""