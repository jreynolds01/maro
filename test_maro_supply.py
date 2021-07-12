from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ManufactureAction, ConsumerAction

from pprint import pprint
import numpy as np


def recursive_read_dict(d, steps):
    for step in steps:
        if d is None or step not in d:
            return None
        d = d[step]
    return d


################################################################################################################
env = Env(scenario='supply_chain', topology='sample', start_tick=0, durations=100)
with open('debug.txt', 'w') as f:
    pprint(env.summary, stream=f)

################################################################################################################
storage_nodes = env.snapshot_list["storage"]
manufacture_nodes = env.snapshot_list["manufacture"]

storage_features = ("id", "facility_id", "capacity", "remaining_space")
manufacture_features = ("id", "facility_id", "manufacturing_number", "product_id", "product_unit_cost")


def show():
    for facility_id, facility_info in env.summary["node_mapping"]["facilities"].items():
        storage_index = facility_info["units"]["storage"]["node_index"]
        storage_states = storage_nodes[env.frame_index:storage_index:storage_features] \
            .flatten() \
            .astype(np.int)
        print(facility_id, facility_info["name"], dict(zip(storage_features, storage_states)))


################################################################################################################
env.step(None)
show()


################################################################################################################
actions = []
actions.append(ManufactureAction(id=7, production_rate=1000))  # sup1 produce sku3
actions.append(ManufactureAction(id=14, production_rate=100))  # sup2 produce sku1
actions.append(ConsumerAction(16, 3, 1, quantity=300, vlt=5, reward_discount=0))  # sku3, sup1 => sup2
actions.append(ConsumerAction(23, 1, 8, quantity=300, vlt=5, reward_discount=0))  # sku1, sup2 => ware
actions.append(ConsumerAction(27, 3, 1, quantity=300, vlt=5, reward_discount=0))  # sku3, sup1 => ware
actions.append(ConsumerAction(31, 1, 17, quantity=300, vlt=5, reward_discount=0))  # sku1, ware => retail
actions.append(ConsumerAction(34, 3, 17, quantity=300, vlt=5, reward_discount=0))  # sku3, ware => retail

for tick in range(10):
    print('\n' + '#' * 20 + f' tick {tick + 1} ' + '#' * 20)
    env.step({action.id: action for action in actions})
    show()
