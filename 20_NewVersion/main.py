from policy_mining import PolicyMining
import sys

NAME_DATASET = sys.argv[1] # AMZ, HC, IoT
ENV_PC = sys.argv[2] # M, S, L
NETWORK_MODEL_GENERATION = False if sys.argv[3] == '0' else True # 0: No generation (False), 1: Generation (True)
FILE_NAME = None
USER_ATTRS = None
RESOURCE_ATTRS = None
USER_NETWORK_PATH = None
USER_NETWORK = ""

# Settings for each dataset
if NAME_DATASET == "AMZ":
    if ENV_PC == "M":
        FILE_NAME = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    elif ENV_PC == "S":
        FILE_NAME = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    elif ENV_PC == "L":
        FILE_NAME = "/home/daniel/Documents/phd/phd-thesis-lab/12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    else:
        print("Invalid enviroment. 'M' for Mac, 'L' for Linux, 'S' for Server")
        
    USER_ATTRS = ["MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME",
                  "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"]
    RESOURCE_ATTRS = ["RID"]

    USER_NETWORK_PATH = ""

    th_rule_sim = 1

elif NAME_DATASET == "HC":
    if ENV_PC == "M":
        FILE_NAME = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/02-HC/01-DistributionsCSV/HC-MOD.csv"
    elif ENV_PC == "S":
        FILE_NAME = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    elif ENV_PC == "L":
        FILE_NAME = "/home/daniel/Documents/phd/phd-thesis-lab/12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    else:
        print("Invalid enviroment. 'M' for Mac, 'L' for Linux, 'S' for Server")
    USER_ATTRS = ['role', 'specialty', 'team', 'uward', 'agentfor']
    RESOURCE_ATTRS = ['type', 'patient', 'treatingteam',
                      'oward', 'author', 'topic']
    
    USER_NETWORK_PATH = ""

    th_rule_sim = 1

elif NAME_DATASET == "IoT":
    if ENV_PC == "M":
        FILE_NAME = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/11-IoT/01-DistributionsCSV/IoT-Rw.csv"
    elif ENV_PC == "S":
        FILE_NAME = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    elif ENV_PC == "L":
        FILE_NAME = "/home/daniel/Documents/phd/phd-thesis-lab/12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-MOD.csv"
    else:
        print("Invalid enviroment. 'M' for Mac, 'L' for Linux, 'S' for Server")
    USER_ATTRS = ["role", "age", "health"]
    RESOURCE_ATTRS = ["type", "area", "mode", "temperature", "lockstatus"]
    USER_NETWORK_PATH = ""
    th_rule_sim = 1
    
else:
    print("Invalid Dataset")
    sys.exit()

if not NETWORK_MODEL_GENERATION:
    # If user not select our projection methodology, we checked the user network
    # pat file.
    try:
        with open(USER_NETWORK_PATH) as file:
            print("User Network File Founded!")
    except FileNotFoundError:
        print("Error: The User Network File was not found at the specified path.")

pm = PolicyMining(FILE_NAME, NAME_DATASET, USER_ATTRS, RESOURCE_ATTRS)
pm.data_preprocessing()
pm.network_model(NETWORK_MODEL_GENERATION, USER_NETWORK_PATH)
pm.community_detection()
pm.rule_inference(th_rule_sim)
pm.evaluation()
pm.policy_refinement(th_rule_sim)
