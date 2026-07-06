from policy_mining import PolicyMining, introducir_ruido, crear_version_sparse
import sys

NAME_DATASET = sys.argv[1]
OS_TYPE = sys.argv[2]
FILE_NAME = None
USER_ATTRS = None
RESOURCE_ATTRS = None
th_rule_sim = None

if (OS_TYPE == "MAC"):
    FILE_NAME = "/Users/ddiaz/Documents/code/"
else:
    FILE_NAME = "/home/daniel/Documents/phd/"

# Settings for each dataset
if NAME_DATASET == "AMZ":
    FILE_NAME = FILE_NAME + "12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    USER_ATTRS = ["MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME",
                  "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"]
    RESOURCE_ATTRS = ["RID"]
    th_rule_sim = 1
    th_big_com = 0.5
    th_med_com = 0.25
    th_lfr = 0.2
    th_sr = 0.5

elif NAME_DATASET == "HC":
    #FILE_NAME = "phd-thesis-lab/12-third_year/00-Data/02-HC/01-DistributionsCSV/HC-MOD.csv"
    FILE_NAME = FILE_NAME + "phd-thesis-lab/12-third_year/00-Data/02-HC/00-Original/0-HC-universal.csv"
    USER_ATTRS = ['position', 'specialties', 'teams', 'uward', 'agentfor']
    RESOURCE_ATTRS = ['type', 'patient', 'tratingTeam',
                      'rward', 'author', 'topics']
    th_rule_sim = 2
    th_big_com = 0.8
    th_med_com = 0.5
    th_lfr = 0.4
    th_sr = 0.6

elif NAME_DATASET == "IoT":
    FILE_NAME = FILE_NAME + "phd-thesis-lab/12-third_year/00-Data/11-IoT/01-DistributionsCSV/IoT-Rw.csv"
    USER_ATTRS = ["role", "age", "health"]
    RESOURCE_ATTRS = ["type", "area", "mode", "temperature", "lockstatus"]
    th_rule_sim = 1
    th_big_com = 0.5
    th_med_com = 0.25
    th_lfr = 0.2
    th_sr = 0.5
    
else:
    print("Invalid Dataset")
    sys.exit()


pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing(False, False)
pm.network_model()
pm.community_detection()
#pm.rule_inference(th_rule_sim, th_lfr, th_med_com)
pm.rule_inference_hc(th_rule_sim, th_lfr, th_med_com)
pm.evaluation()
res1 = pm.policy_refinement(th_rule_sim)