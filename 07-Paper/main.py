from policy_mining import PolicyMining
import sys

NAME_DATASET = sys.argv[1]
FILE_NAME = None
USER_ATTRS = None
RESOURCE_ATTRS = None
th_rule_sim = None

# Settings for each dataset
if NAME_DATASET == "AMZ":
    FILE_NAME = "../00-Data/train.csv"
    USER_ATTRS = ["MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME",
                  "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"]
    RESOURCE_ATTRS = ["RID"]
    th_rule_sim = 1

elif NAME_DATASET == "HC":
    FILE_NAME = "../00-Data/HC-AccessLog.csv"
    USER_ATTRS = ['role', 'specialty', 'team', 'uward', 'agentfor']
    RESOURCE_ATTRS = ['type', 'patient', 'treatingteam',
                      'oward', 'author', 'topic']
    th_rule_sim = 1

elif NAME_DATASET == "CAV":
    FILE_NAME = "../00-Data/cav_policies.csv"
    USER_ATTRS = ["monitoring", "fallback",
                  "weather", "visibility", "traffic_congestion"]
    RESOURCE_ATTRS = ["driving_task_loa", "vehicle_loa", "region_loa"]
    th_rule_sim = 1
else:
    print("Invalid Dataset")
    sys.exit()

pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing()
pm.network_model()
pm.community_detection()
pm.rule_inference(th_rule_sim)
pm.evaluation()
pm.policy_refinement(th_rule_sim)
